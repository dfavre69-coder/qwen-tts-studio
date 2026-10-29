# Copyright 2026 Denis Favre
# SPDX-License-Identifier: Apache-2.0
#
# Le tirage (température, top-k, top-p) reproduit à l'identique la logique
# de Hugging Face transformers (Apache 2.0) ; le module s'appuie sur les
# classes du paquet qwen-tts d'Alibaba Qwen (Apache 2.0) sans les modifier.
# =============================================================================
#  ACCÉLÉRATION DU SOUS-MODÈLE QWEN3-TTS (« code predictor »)
# =============================================================================
# Pour chaque trame audio (1/12 s), Qwen3-TTS appelle un petit sous-modèle
# qui produit 15 codes l'un après l'autre. Via transformers.generate(), cela
# représente ~2 500 petites opérations GPU lancées une à une par le
# processeur : le GPU passe l'essentiel de son temps à attendre.
#
# Deux niveaux :
#   1. boucle directe  : même calcul que generate(), sans sa lourde
#      préparation (résultat identique au bit près) ;
#   2. CUDA Graph      : la séquence complète des 15 codes est enregistrée
#      une fois par taille de lot, puis rejouée en un seul lancement.
#      Contrôlée au démarrage contre la boucle directe ; en cas d'écart ou
#      d'erreur, retour automatique au niveau 1.
# =============================================================================

import types


class _CacheFixe:
    """Cache clé/valeur préalloué (tailles fixes, compatible CUDA Graph).
    Implémente la seule méthode utilisée par les couches d'attention."""

    def __init__(self, torch, n_couches, lot, tetes_kv, longueur, dim, dtype, device):
        self.k = [torch.zeros(lot, tetes_kv, longueur, dim, dtype=dtype, device=device)
                  for _ in range(n_couches)]
        self.v = [torch.zeros_like(t) for t in self.k]
        self.pos = 0

    def update(self, cle, valeur, indice_couche, _kwargs=None):
        t = cle.shape[2]
        p = self.pos
        self.k[indice_couche][:, :, p:p + t].copy_(cle)
        self.v[indice_couche][:, :, p:p + t].copy_(valeur)
        return self.k[indice_couche][:, :, :p + t], self.v[indice_couche][:, :, :p + t]


def decoder_par_groupes(qwen_model, taille=1):
    """Fait décoder l'audio par petits groupes au lieu du lot entier.
    Le décodage redécode la voix de référence avec CHAQUE phrase : en lot de
    8, il occupait ~6 Go de VRAM, contre ~0,75 Go phrase par phrase, pour le
    même temps de calcul. Renvoie True si installé."""
    tok = getattr(getattr(qwen_model, "model", None), "speech_tokenizer", None)
    if tok is None or getattr(tok, "_decodage_origine", None) is not None:
        return tok is not None
    origine = tok.decode

    def decode_par_groupes(encoded, *a, **k):
        if not isinstance(encoded, list) or len(encoded) <= taille:
            return origine(encoded, *a, **k)
        wavs, fs = [], None
        for i in range(0, len(encoded), taille):
            w, fs = origine(encoded[i:i + taille], *a, **k)
            wavs.extend(w)
        return wavs, fs

    tok.decode = decode_par_groupes
    tok._decodage_origine = origine
    return True


def installer_acceleration(qwen_model, torch, graphes=True, journal=print,
                           decodage_par=1):
    """Installe l'accélération sur un Qwen3TTSModel chargé. Renvoie un
    libellé décrivant le niveau obtenu. decodage_par=0 laisse le décodage
    audio d'origine (tout le lot d'un coup)."""
    from transformers.cache_utils import DynamicCache

    if decodage_par and decodage_par > 0:
        decoder_par_groupes(qwen_model, decodage_par)

    talker = getattr(getattr(qwen_model, "model", None), "talker", None)
    cp = getattr(talker, "code_predictor", None)
    if cp is None:
        return "indisponible (structure de modèle inattendue)"
    if getattr(cp, "_acceleration", None):
        return cp._acceleration

    cfg = cp.config
    n_codes = cfg.num_code_groups - 1
    embeddings = cp.model.get_input_embeddings()
    projection = cp.small_to_mtp_projection
    couches = list(cp.model.layers[: cfg.num_hidden_layers])
    attn0 = couches[0].self_attn
    eager = getattr(cfg, "_attn_implementation", "sdpa") == "eager"
    origine = cp.generate

    # ------------------------------------------------------------- tirage
    def tirer(logits, do_sample, top_k, top_p, temperature):
        """Température -> top-k -> top-p -> tirage, comme transformers.
        Le tirage par loi exponentielle est exactement celui qu'emploie
        torch.multinomial pour un seul échantillon, mais sans
        synchronisation : il est donc enregistrable dans un CUDA Graph."""
        if not do_sample:
            return logits.argmax(dim=-1)
        if temperature is not None and temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and top_k != 0:
            k = min(int(top_k), logits.size(-1))
            seuil = torch.topk(logits, k)[0][..., -1, None]
            logits = logits.masked_fill(logits < seuil, -float("inf"))
        if top_p is not None and top_p < 1.0:
            tries, indices = torch.sort(logits, descending=False)
            cumul = tries.softmax(dim=-1).cumsum(dim=-1)
            retirer = cumul <= (1 - top_p)
            retirer[..., -1:] = 0
            retirer = retirer.scatter(1, indices, retirer)
            logits = logits.masked_fill(retirer, -float("inf"))
        probas = torch.nn.functional.softmax(logits, dim=-1)
        q = torch.empty_like(probas).exponential_(1)
        return (probas / q).argmax(dim=-1)

    # ------------------------------------------- niveau 1 : boucle directe
    @torch.no_grad()
    def boucle_directe(inputs_embeds, n, do_sample, top_k, top_p, temperature):
        cache = DynamicCache()
        sortie = cp.model(inputs_embeds=projection(inputs_embeds),
                          past_key_values=cache, use_cache=True)
        etat = sortie.last_hidden_state[:, -1, :]
        codes = []
        for g in range(n):
            logits = cp.lm_head[g](etat).to(dtype=torch.float32)
            jeton = tirer(logits, do_sample, top_k, top_p, temperature)
            codes.append(jeton)
            if g == n - 1:
                break
            e = projection(embeddings[g](jeton[:, None]))
            sortie = cp.model(inputs_embeds=e, past_key_values=cache, use_cache=True)
            etat = sortie.last_hidden_state[:, -1, :]
        return torch.stack(codes, dim=1)

    # ------------------------- niveau 2 : passage à tailles fixes (graphe)
    longueur_max = n_codes + 1                   # 2 entrées + 14 codes réinjectés
    tetes_kv = cfg.num_key_value_heads
    dim_tete = attn0.head_dim

    def preparer_positions(device, dtype):
        """cos/sin et positions de chaque pas, calculés hors graphe."""
        pos = []
        factice = torch.zeros(1, 1, cfg.hidden_size, device=device, dtype=dtype)
        for debut, t in [(0, 2)] + [(2 + i, 1) for i in range(n_codes - 1)]:
            cache_pos = torch.arange(debut, debut + t, device=device)
            ids = cache_pos.unsqueeze(0)
            cos, sin = cp.model.rotary_emb(factice, ids)
            masque = None
            if t > 1 and eager:
                masque = torch.full((t, t), torch.finfo(dtype).min, device=device, dtype=dtype)
                masque = torch.triu(masque, diagonal=1)[None, None]
            pos.append((cache_pos, ids, (cos, sin), masque))
        return pos

    def passe(x, cache, p):
        cache_pos, ids, rot, masque = p
        h = x
        for couche in couches:
            h = couche(h, attention_mask=masque, position_ids=ids,
                       past_key_values=cache, use_cache=True,
                       cache_position=cache_pos, position_embeddings=rot)
            h = h[0] if isinstance(h, tuple) else h
        cache.pos += x.shape[1]
        return cp.model.norm(h)[:, -1, :]

    def sequence_fixe(entree, cache, positions, do_sample, top_k, top_p, temperature):
        cache.pos = 0
        etat = passe(projection(entree), cache, positions[0])
        codes = []
        for g in range(n_codes):
            logits = cp.lm_head[g](etat).to(dtype=torch.float32)
            jeton = tirer(logits, do_sample, top_k, top_p, temperature)
            codes.append(jeton)
            if g == n_codes - 1:
                break
            e = projection(embeddings[g](jeton[:, None]))
            etat = passe(e, cache, positions[g + 1])
        return torch.stack(codes, dim=1)

    etat_graphes = {"actif": False, "graphes": {}, "positions": None}

    def obtenir_graphe(lot, params, device, dtype):
        cle = (lot,) + params
        g = etat_graphes["graphes"].get(cle)
        if g is not None:
            return g
        if len(etat_graphes["graphes"]) >= 8:          # borne la mémoire
            etat_graphes["graphes"].pop(next(iter(etat_graphes["graphes"])))
        if etat_graphes["positions"] is None:
            etat_graphes["positions"] = preparer_positions(device, dtype)
        positions = etat_graphes["positions"]
        entree = torch.zeros(lot, 2, talker.config.hidden_size, device=device, dtype=dtype)
        cache = _CacheFixe(torch, len(couches), lot, tetes_kv, longueur_max,
                           dim_tete, dtype, device)
        flux = torch.cuda.Stream()
        flux.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(flux), torch.no_grad():
            for _ in range(2):                         # préchauffage obligatoire
                sequence_fixe(entree, cache, positions, *params)
        torch.cuda.current_stream().wait_stream(flux)
        graphe = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphe), torch.no_grad():
            sortie = sequence_fixe(entree, cache, positions, *params)
        # TOUT ce que le graphe lit ou écrit doit rester en vie aussi longtemps
        # que lui : s'il était libéré, sa mémoire serait réattribuée au modèle
        # principal et chaque rejeu l'écraserait.
        g = {"graphe": graphe, "entree": entree, "sortie": sortie,
             "cache": cache, "positions": positions, "verifs": 0}
        etat_graphes["graphes"][cle] = g
        return g

    vocab = cfg.vocab_size

    def via_graphe(inputs_embeds, params):
        g = obtenir_graphe(
            inputs_embeds.shape[0], params, inputs_embeds.device, inputs_embeds.dtype)
        g["entree"].copy_(inputs_embeds)
        g["graphe"].replay()
        codes = g["sortie"].clone()
        # Garde-fou : les premiers rejeux de chaque graphe sont vérifiés
        # (codes dans le vocabulaire) avant de lui faire confiance.
        if g["verifs"] < 20:
            g["verifs"] += 1
            if int(codes.min()) < 0 or int(codes.max()) >= vocab:
                raise RuntimeError("codes hors vocabulaire en sortie du graphe")
        return codes

    # --------------------------------------------------- point d'entrée
    @torch.no_grad()
    def generate_accelere(inputs_embeds=None, max_new_tokens=None, do_sample=True,
                          top_p=1.0, top_k=50, temperature=1.0, **_ignores):
        n = n_codes if max_new_tokens is None else int(max_new_tokens)
        params = (bool(do_sample), top_k, top_p, temperature)
        if (etat_graphes["actif"] and n == n_codes
                and inputs_embeds is not None and inputs_embeds.shape[1] == 2):
            try:
                return types.SimpleNamespace(sequences=via_graphe(inputs_embeds, params))
            except Exception as e:  # noqa: BLE001
                etat_graphes["actif"] = False
                etat_graphes["graphes"].clear()
                journal(f"⚠ CUDA Graph désactivé en cours de route ({e}) — "
                        "retour à la boucle directe.")
        return types.SimpleNamespace(
            sequences=boucle_directe(inputs_embeds, n, *params))

    cp.generate = generate_accelere
    cp._generate_origine = origine
    cp._acceleration_interne = {          # pour les tests hors GPU
        "sequence_fixe": sequence_fixe, "preparer_positions": preparer_positions,
        "cache": lambda lot, device, dtype: _CacheFixe(
            torch, len(couches), lot, tetes_kv, longueur_max, dim_tete, dtype, device),
        "boucle_directe": boucle_directe,
    }
    libelle = "boucle directe"

    # ------------------------------------------ contrôle avant activation
    try:
        param = next(cp.parameters())
        device, dtype = param.device, param.dtype
        if graphes and device.type == "cuda":
            # Contrôle en deux temps, en mode glouton (sans hasard) :
            #  1. le graphe rejoue EXACTEMENT le même calcul exécuté normalement ;
            #  2. ce calcul concorde avec la bibliothèque d'origine (de minimes
            #     écarts d'arrondi fp16 entre noyaux d'attention sont tolérés ;
            #     un graphe défectueux donnerait des codes quelconques).
            essai = torch.randn(4, 2, talker.config.hidden_size, device=device, dtype=dtype)
            glouton = (False, 50, 1.0, 1.0)
            with torch.no_grad():
                reference = boucle_directe(essai, n_codes, *glouton)
                normal = sequence_fixe(
                    essai,
                    _CacheFixe(torch, len(couches), 4, tetes_kv, longueur_max,
                               dim_tete, dtype, device),
                    preparer_positions(device, dtype), *glouton)
            etat_graphes["actif"] = True
            obtenu = via_graphe(essai, glouton)
            # Occupe la mémoire libre, rejoue, puis vérifie que le graphe
            # n'a écrit que dans SES tampons.
            # (petits blocs : c'est là que l'allocateur recycle les tampons libérés)
            bourrage = [torch.full((32768,), 7.0, device=device) for _ in range(512)]
            obtenu2 = via_graphe(essai, glouton)
            torch.cuda.synchronize()
            intact = all(bool((t == 7.0).all()) for t in bourrage)
            del bourrage
            fidele = intact and torch.equal(normal, obtenu) and torch.equal(normal, obtenu2)
            accord = (reference == normal).float().mean().item()
            if fidele and accord >= 0.9:
                libelle = f"CUDA Graph (contrôlé, {accord:.0%} de concordance)"
            else:
                etat_graphes["actif"] = False
                etat_graphes["graphes"].clear()
                raison = (("mémoire étrangère modifiée" if not intact else "rejeu non fidèle")
                          if not fidele
                          else f"{accord:.0%} de concordance seulement")
                libelle = f"boucle directe (CUDA Graph écarté : {raison})"
        elif graphes:
            libelle = "boucle directe (CUDA Graph : GPU requis)"
    except Exception as e:  # noqa: BLE001
        etat_graphes["actif"] = False
        etat_graphes["graphes"].clear()
        libelle = f"boucle directe (CUDA Graph indisponible : {e})"

    cp._acceleration = libelle
    return libelle


def retirer_acceleration(qwen_model):
    cp = qwen_model.model.talker.code_predictor
    if getattr(cp, "_acceleration", None):
        cp.generate = cp._generate_origine
        cp._acceleration = None
    tok = getattr(qwen_model.model, "speech_tokenizer", None)
    if tok is not None and getattr(tok, "_decodage_origine", None) is not None:
        tok.decode = tok._decodage_origine
        tok._decodage_origine = None
