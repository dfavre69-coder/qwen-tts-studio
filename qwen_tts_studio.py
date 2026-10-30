# -*- coding: utf-8 -*-
"""
================================================================================
  QWEN3-TTS VOICE STUDIO
  Application de bureau (Windows) de clonage vocal et de lecture de textes,
  construite sur le modèle libre Qwen3-TTS.
================================================================================

  Copyright 2026 Denis Favre
  SPDX-License-Identifier: Apache-2.0
  Distribué sous licence Apache 2.0 — voir les fichiers LICENSE et NOTICE.

  FONCTIONS
  ---------
    • Voix de référence (WAV / MP3 / FLAC) + sa transcription → voix clonée
    • Texte source : .txt, .md et .docx (Word)
    • Nettoyage automatique du texte (Markdown, emojis, nombres et chiffres
      romains écrits en toutes lettres, symboles)
    • Découpage en morceaux et génération PAR LOTS sur GPU, accélérée par
      acceleration_qwen.py (CUDA Graph) — environ 7 s d'audio par seconde
      de calcul sur une RTX A4500
    • Export WAV, rééchantillonnage 44,1 / 48 kHz, export MP3
    • Transcription automatique de la voix de référence (Whisper)
    • Contrôle qualité : Whisper réécoute chaque phrase et fait régénérer
      celles qui ne correspondent pas au texte
    • Sous-titres .srt synchronisés
    • Dialogues à plusieurs voix : balises [Nom] en début de paragraphe
    • Livre audio M4B : un chapitre par titre du document, couverture,
      titre et auteur
    • Correction : régénération de quelques phrases sans tout relancer
    • Montage vidéo : image fixe + piste audio → MP4 / MKV / MOV
    • Fusion de plusieurs fichiers audio avec chapitrage automatique
      (description YouTube, fichier texte, chapitres intégrés)

  DÉPENDANCES
  -----------
    Voir requirements.txt et requirements-torch.txt (installer.bat s'en
    charge). ffmpeg doit être installé pour le MP3, la vidéo et la fusion.

  LANCEMENT
  ---------
        lancer_studio.bat        (ou : qwen_env\\Scripts\\python.exe qwen_tts_studio.py)

  L'interface s'ouvre instantanément : torch et qwen_tts ne sont chargés
  qu'au premier rendu (imports différés), pour ne pas attendre au démarrage.

  ⚠ N'utilisez une voix qu'avec le consentement de la personne concernée.
================================================================================
"""

import os
import re
import gc
import glob
import json
import time
import queue
import shutil
import threading
import subprocess
from math import gcd
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# =============================================================================
#  RÉGLAGE RAPIDE
# =============================================================================
# Nombre de phrases générées en parallèle sur le GPU (taille de lot).
#   1      = une phrase à la fois (ancien comportement, le plus lent)
#   32     = mesuré sur la RTX A4500 (20 Go) : 7,4 s d'audio par seconde
#            pour 11,8 Go de VRAM (audio décodé phrase par phrase).
# En cas de débordement, le studio réduit de lui-même le lot en cours de rendu.
# Valeur appliquée à CHAQUE démarrage du studio : elle prime sur celle
# enregistrée dans la configuration. Le champ de l'onglet « Voix &
# Découpage » permet encore de la changer le temps d'une session.
TAILLE_LOT = 32

# Accélération du sous-modèle (boucle directe + CUDA Graph contrôlé).
# Nécessite acceleration_qwen.py dans le même dossier que ce fichier.
# False = comportement d'origine de la bibliothèque Qwen3-TTS.
ACCELERATION = True

# Les bibliothèques lourdes (torch, qwen_tts, soundfile, numpy, scipy) sont
# importées via precharger_modules(), appelé UNE SEULE FOIS au démarrage
# depuis le THREAD PRINCIPAL.
#
# ⚠️  Point critique Windows : le module natif d'onnxruntime (tiré par
# qwen_tts) possède une routine d'initialisation DLL qui ÉCHOUE si le tout
# premier `import` a lieu sur un thread secondaire (erreur 1114,
# « DLL load failed while importing onnxruntime_pybind11_state »).
# On force donc l'import sur le thread principal au lancement ; le thread de
# travail ne fait ensuite que RÉUTILISER des modules déjà chargés en mémoire.

_MODULES = {}


def precharger_modules():
    """Importe les bibliothèques natives et les met en cache. DOIT être appelé
    la première fois depuis le thread principal (voir note Windows ci-dessus).
    Les appels suivants — y compris depuis le thread de travail — renvoient
    simplement le cache, sans réimport natif."""
    if _MODULES.get("ok"):
        return _MODULES
    import torch
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from qwen_tts import Qwen3TTSModel  # déclenche l'import d'onnxruntime
    _MODULES.update(
        torch=torch, np=np, sf=sf, resample_poly=resample_poly,
        Qwen3TTSModel=Qwen3TTSModel, ok=True,
    )
    return _MODULES


# =============================================================================
#  SECTION 1 — TRAITEMENT DU TEXTE  (logique reprise intégralement du script)
# =============================================================================

ROMAINS = {
    'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
    'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
    'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1,
}


def romain_vers_arabe(match):
    """Convertit un chiffre romain isolé en nombre arabe (texte)."""
    s = match.group(0)
    resultat = 0
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i + 2] in ROMAINS:
            resultat += ROMAINS[s[i:i + 2]]
            i += 2
        elif s[i] in ROMAINS:
            resultat += ROMAINS[s[i]]
            i += 1
        else:
            return s
    return str(resultat)


def nombre_en_lettres_fr(n):
    if n == 0:
        return "zéro"

    unites = [
        "", "un", "deux", "trois", "quatre", "cinq", "six", "sept",
        "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze",
        "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf"
    ]
    dizaines = [
        "", "", "vingt", "trente", "quarante", "cinquante",
        "soixante", "soixante", "quatre-vingt", "quatre-vingt"
    ]

    def _deux_chiffres(n):
        if n < 20:
            return unites[n]
        d, u = divmod(n, 10)
        if d == 7:
            return "soixante-" + ("et-onze" if u == 1 else unites[10 + u])
        if d == 9:
            return "quatre-vingt-" + unites[10 + u]
        if u == 0:
            return dizaines[d] + ("s" if d == 8 else "")
        liaison = "-et-" if u == 1 and d in (2, 3, 4, 5, 6) else "-"
        return dizaines[d] + liaison + unites[u]

    def _trois_chiffres(n):
        c, r = divmod(n, 100)
        txt = ""
        if c == 1:
            txt = "cent"
        elif c > 1:
            txt = unites[c] + " cent"
        if r == 0:
            return txt + ("s" if c > 1 else "")
        return (txt + " " + _deux_chiffres(r)).strip()

    parties = []
    if n >= 1000:
        milliers = n // 1000
        reste = n % 1000
        if milliers == 1:
            parties.append("mille")
        else:
            parties.append(_trois_chiffres(milliers) + " mille")
        if reste:
            parties.append(_trois_chiffres(reste))
    else:
        parties.append(_trois_chiffres(n))

    return " ".join(parties)


def chiffres_vers_lettres(match):
    texte_nombre = match.group(0).replace(" ", "").replace("\u202f", "")
    try:
        n = int(texte_nombre)
        if 0 <= n <= 999999:
            return nombre_en_lettres_fr(n)
        return texte_nombre
    except ValueError:
        return texte_nombre


def nettoyer_texte_pour_audio(texte):
    """Supprime tout ce qui ne doit pas être lu à voix haute (Markdown, emojis,
    symboles) et translittère nombres et chiffres romains en toutes lettres."""
    texte = re.sub(r'^#{1,6}\s*', '', texte, flags=re.MULTILINE)
    texte = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', texte)
    texte = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', texte)
    texte = re.sub(r'~~(.+?)~~', r'\1', texte)
    texte = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', texte)
    texte = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', texte)
    texte = re.sub(r'```[\s\S]*?```', '', texte)
    texte = re.sub(r'`([^`]+)`', r'\1', texte)
    texte = re.sub(r'^[\s]*[-*_]{3,}[\s]*$', '', texte, flags=re.MULTILINE)
    texte = re.sub(r'^\s*[-*+]\s+', '', texte, flags=re.MULTILINE)
    texte = re.sub(r'^\s*\d+\.\s+', '', texte, flags=re.MULTILINE)
    texte = re.sub(r'^\s*>\s?', '', texte, flags=re.MULTILINE)
    texte = re.sub(r'<[^>]+>', '', texte)
    texte = re.sub(r'^[\s]*[*=~#\-_]{2,}[\s]*$', '', texte, flags=re.MULTILINE)
    texte = re.sub(
        r'[\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F900-\U0001F9FF'
        r'\U00002702-\U000027B0'
        r'\U0001FA00-\U0001FA6F'
        r'\U0001FA70-\U0001FAFF'
        r'\U00002600-\U000026FF'
        r'\U0000FE00-\U0000FE0F'
        r'\U0000200D'
        r'\U00002B50'
        r'\U0000203C-\U00003299]+', '', texte
    )
    texte = re.sub(r'\b([IVXLCDM]{2,})\b', romain_vers_arabe, texte)
    texte = re.sub(
        r'(?i)\b(chapitre|partie|acte|tome|volume|section|livre|épisode|saison|phase|étape|point|n°|numéro)\s+([IVXLCDM])\b',
        lambda m: m.group(1) + " " + romain_vers_arabe(m), texte
    )
    texte = re.sub(
        r'(?i)\b(chapitre|partie|acte|tome|volume|section|livre|épisode|saison|phase|étape|point|n°|numéro)\s+([IVXLCDM])\b',
        lambda m: m.group(1) + " " + str(
            {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}.get(m.group(2).upper(), m.group(2))
        ), texte
    )
    texte = re.sub(r'\b(\d{1,3}(?:[\s\u202f]\d{3})+)\b', chiffres_vers_lettres, texte)
    texte = re.sub(r'\b(\d{1,6})\b', chiffres_vers_lettres, texte)
    texte = texte.replace('%', ' pour cent')
    texte = texte.replace('€', ' euros')
    texte = texte.replace('$', ' dollars')
    texte = texte.replace('£', ' livres sterling')
    texte = texte.replace('&', ' et ')
    texte = texte.replace('+', ' plus ')
    texte = texte.replace('=', ' égale ')
    texte = texte.replace('@', ' arobase ')
    texte = texte.replace('°C', ' degrés Celsius')
    texte = texte.replace('°F', ' degrés Fahrenheit')
    texte = texte.replace('°', ' degrés ')
    texte = texte.replace('²', ' au carré')
    texte = texte.replace('³', ' au cube')
    texte = texte.replace('/', ' sur ')
    # On préserve l'apostrophe droite pour les contractions françaises.
    texte = re.sub(r'[«»""""\[\]{}()⟨⟩]', ' ', texte)
    texte = re.sub(r'\.{2,}', '.', texte)
    texte = re.sub(r'[-–—]{2,}', ', ', texte)
    texte = re.sub(r'[ \t]+', ' ', texte)
    texte = re.sub(r'\n\s*\n+', '\n', texte)
    texte = re.sub(r'^\s+|\s+$', '', texte, flags=re.MULTILINE)
    return texte.strip()


def decouper_en_chunks(texte, max_chars):
    """Découpage hiérarchique : phrases > propositions > mots."""
    texte = texte.strip()
    if not texte:
        return []
    if len(texte) <= max_chars:
        return [texte]

    chunks = []

    sous_phrases = re.split(r'(?<=[.!?;:])\s+', texte)
    if len(sous_phrases) > 1:
        for sp in sous_phrases:
            chunks.extend(decouper_en_chunks(sp, max_chars))
        return chunks

    sous_phrases = re.split(r'(?<=[,\-–—])\s+', texte)
    if len(sous_phrases) > 1:
        tampon = ""
        for sp in sous_phrases:
            candidat = (tampon + " " + sp).strip() if tampon else sp
            if len(candidat) <= max_chars:
                tampon = candidat
            else:
                if tampon:
                    chunks.extend(decouper_en_chunks(tampon, max_chars))
                tampon = sp
        if tampon:
            chunks.extend(decouper_en_chunks(tampon, max_chars))
        return chunks

    mots = texte.split()
    tampon = ""
    for mot in mots:
        candidat = (tampon + " " + mot).strip() if tampon else mot
        if len(candidat) <= max_chars:
            tampon = candidat
        else:
            if tampon:
                chunks.append(tampon)
            tampon = mot
    if tampon:
        chunks.append(tampon)

    return chunks


def construire_chunks(texte_brut, max_chars):
    """Nettoie le texte, marque les paragraphes et renvoie la liste finale
    (chunk, est_fin_paragraphe) ainsi que des statistiques. Utilisé à la fois
    par l'aperçu (sans modèle) et par la génération."""
    texte_long = nettoyer_texte_pour_audio(texte_brut)

    MARQUEUR = "§§FINPARA§§"
    texte_long = re.sub(r'\n+', f' {MARQUEUR} ', texte_long)

    phrases_brutes = re.split(r'(?<=[.!?;:>])\s+', texte_long.strip())
    phrases_brutes = [p.strip() for p in phrases_brutes if len(p.strip()) > 2]

    chunks_finaux = []
    for phrase in phrases_brutes:
        est_fin_para = MARQUEUR in phrase
        phrase_propre = phrase.replace(MARQUEUR, "").strip()
        if len(phrase_propre) <= 2:
            continue
        morceaux = decouper_en_chunks(phrase_propre, max_chars)
        for j, m in enumerate(morceaux):
            derniere = (j == len(morceaux) - 1) and est_fin_para
            chunks_finaux.append((m, derniere))

    chunks_finaux = [(c, fp) for (c, fp) in chunks_finaux if len(c.strip()) > 2]

    tailles = [len(c) for c, _ in chunks_finaux] or [0]
    stats = {
        "nb_chunks": len(chunks_finaux),
        "min": min(tailles),
        "max": max(tailles),
        "moyenne": sum(tailles) // max(len(tailles), 1),
        "nb_fin_para": sum(1 for _, fp in chunks_finaux if fp),
        "texte_nettoye": nettoyer_texte_pour_audio(texte_brut),
    }
    return chunks_finaux, stats


# =============================================================================
#  SECTION 2 — UTILITAIRES AUDIO
# =============================================================================

def normaliser_peak(audio, target_dbfs):
    """Normalise un signal au niveau de crête cible (peak normalization)."""
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        return audio
    target_linear = 10.0 ** (target_dbfs / 20.0)
    gain = target_linear / peak
    return (audio * gain).astype(np.float32)


def duree_silence_apres(chunk, est_fin_paragraphe, sil_virgule, sil_point, sil_para):
    """Durée du silence à insérer après un chunk selon sa ponctuation finale."""
    if est_fin_paragraphe:
        return sil_para
    dernier = chunk.rstrip()[-1:] if chunk.rstrip() else ""
    if dernier in ".!?;":
        return sil_point
    if dernier in ",-–—:":
        return sil_virgule
    return sil_virgule


# =============================================================================
#  SECTION 3 — LECTURE DES FICHIERS TEXTE (txt / md / docx)
# =============================================================================

def _ponctuer_titre(texte):
    """Ajoute un point final à un titre qui n'en a pas : sans lui, le titre
    serait lu d'un trait avec la phrase suivante."""
    t = texte.strip()
    if t and not re.search(r'[.!?…:;]["»”)\]]*$', t):
        t += "."
    return t


def lire_paragraphes(chemin):
    """Lit un document en conservant sa structure. Renvoie une liste de
    paragraphes {"texte", "niveau", "titre"} : niveau 0 pour un paragraphe
    ordinaire, 1 à 9 pour un titre de ce niveau (styles Word « Titre 1 »,
    « Heading 1 »… ou « # » en Markdown)."""
    ext = os.path.splitext(chemin)[1].lower()

    if ext == ".docx":
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError(
                "Le module python-docx est requis pour lire les fichiers Word.\n"
                "Installe-le avec :  pip install python-docx"
            )
        doc = Document(chemin)
        paragraphes = []
        for p in doc.paragraphs:
            try:
                nom = (p.style.name or "") if p.style is not None else ""
            except Exception:
                nom = ""
            m = re.match(r'(?i)^\s*(heading|titre)\s*(\d)\s*$', nom)
            est_titre = bool(m) or nom.strip().lower() in ("title", "titre")
            paragraphes.append({"texte": p.text,
                                "niveau": int(m.group(2)) if m else 0,
                                "titre": est_titre})
        return paragraphes

    if ext == ".doc":
        raise RuntimeError(
            "L'ancien format .doc n'est pas pris en charge directement.\n"
            "Ouvre le fichier dans Word et enregistre-le au format .docx."
        )

    # .txt, .md, et tout autre format texte brut
    with open(chemin, "r", encoding="utf-8", errors="replace") as f:
        contenu = f.read()
    markdown = ext in (".md", ".markdown")
    paragraphes = []
    for ligne in contenu.splitlines():
        m = re.match(r'^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$', ligne) if markdown else None
        if m:
            paragraphes.append({"texte": m.group(2), "niveau": len(m.group(1)), "titre": True})
        else:
            paragraphes.append({"texte": ligne, "niveau": 0, "titre": False})
    return paragraphes


def lire_fichier_texte(chemin):
    """Lit un fichier texte (.txt, .md, .docx) et le renvoie d'un bloc, un
    paragraphe par ligne. Les titres reçoivent un point final s'il manque."""
    return "\n".join(
        _ponctuer_titre(p["texte"]) if (p["titre"] and p["texte"].strip()) else p["texte"]
        for p in lire_paragraphes(chemin))


# Balise de voix en tête de paragraphe : « [Marie] Bonjour… » ou « [Marie] : … »
BALISE_VOIX = re.compile(r'^\s*\[([^\[\]\n]{1,40})\]\s*[:\-–—]?\s*')
NOMS_NARRATEUR = {"narrateur", "narratrice", "narration", "récit", "recit", "voix principale"}
EXTENSIONS_VOIX = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def construire_plan(paragraphes, max_chars, par_chapitres=False):
    """Prépare le rendu : découpe le texte en chunks et note, pour chacun, la
    voix qui le lit (balise [Nom] en tête de paragraphe ; None = voix
    principale) et le chapitre auquel il appartient (titres du document, si
    par_chapitres). Renvoie un dict : chunks [(texte, fin_para)], voix,
    chapitre, titres, stats."""
    niveaux = [p["niveau"] for p in paragraphes if p["niveau"] > 0 and p["texte"].strip()]
    niveau_chap = min(niveaux) if (par_chapitres and niveaux) else None

    blocs = []          # [chapitre, voix, [textes]]
    titres = []
    chap = -1
    for p in paragraphes:
        t = p["texte"]
        if not t.strip():
            continue
        voix = None
        if niveau_chap and 0 < p["niveau"] <= niveau_chap:
            titres.append(t.strip())
            chap = len(titres) - 1
            texte = _ponctuer_titre(t)
        else:
            if niveau_chap and chap == -1:
                titres.append("Introduction")
                chap = 0
            m = BALISE_VOIX.match(t)
            if m and not p.get("titre"):
                nom = m.group(1).strip()
                voix = None if nom.lower() in NOMS_NARRATEUR else nom
                t = t[m.end():]
            texte = _ponctuer_titre(t) if p.get("titre") else t
        c = max(chap, 0)
        if blocs and blocs[-1][0] == c and blocs[-1][1] == voix:
            blocs[-1][2].append(texte)
        else:
            blocs.append([c, voix, [texte]])

    chunks, voix_l, chap_l, nettoye = [], [], [], []
    for c, voix, textes in blocs:
        morceaux, st = construire_chunks("\n".join(textes), max_chars)
        if not morceaux:
            continue
        morceaux[-1] = (morceaux[-1][0], True)       # fin de bloc = fin de paragraphe
        chunks += morceaux
        voix_l += [voix] * len(morceaux)
        chap_l += [c] * len(morceaux)
        nettoye.append(st["texte_nettoye"])

    utilises = sorted(set(chap_l))
    renum = {ancien: i for i, ancien in enumerate(utilises)}
    chap_l = [renum[c] for c in chap_l]
    titres = [titres[c] if c < len(titres) else "" for c in utilises] if niveau_chap else [""]

    tailles = [len(c) for c, _ in chunks] or [0]
    stats = {
        "nb_chunks": len(chunks),
        "min": min(tailles),
        "max": max(tailles),
        "moyenne": sum(tailles) // max(len(tailles), 1),
        "nb_fin_para": sum(1 for _, fp in chunks if fp),
        "texte_nettoye": "\n".join(nettoye),
        "voix": sorted({v for v in voix_l if v}),
        "nb_chapitres": len(titres) if niveau_chap else 0,
    }
    return {"chunks": chunks, "voix": voix_l, "chapitre": chap_l,
            "titres": titres, "stats": stats}


def trouver_voix(dossier, nom):
    """Cherche « Nom.wav » (ou .flac, .mp3…) dans le dossier des voix, sans
    tenir compte des majuscules. Renvoie (audio, transcription ou None)."""
    if not dossier or not os.path.isdir(dossier):
        return None
    cible = nom.strip().lower()
    for f in sorted(os.listdir(dossier)):
        base, ext = os.path.splitext(f)
        if ext.lower() in EXTENSIONS_VOIX and base.strip().lower() == cible:
            txt = os.path.join(dossier, base + ".txt")
            return os.path.join(dossier, f), (txt if os.path.isfile(txt) else None)
    return None


def trouver_ffmpeg(chemin_manuel=None):
    """Localise l'exécutable ffmpeg, dans cet ordre :
       1) le chemin manuel fourni (un fichier .exe OU un dossier le contenant) ;
       2) le PATH du processus (shutil.which) ;
       3) les emplacements d'installation usuels sous Windows — winget, puis
          Chocolatey, puis installation manuelle classique.
    Renvoie le chemin complet vers ffmpeg(.exe), ou None si introuvable.

    Cette détection évite d'avoir à toucher au PATH système : même si winget
    vient d'installer ffmpeg et que l'application n'a pas hérité du PATH mis à
    jour, on sait le retrouver dans son dossier d'installation."""
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    # 1) Chemin indiqué manuellement par l'utilisateur
    if chemin_manuel:
        p = chemin_manuel.strip().strip('"')
        if p:
            if os.path.isdir(p):
                cand = os.path.join(p, exe)
                if os.path.isfile(cand):
                    return cand
            elif os.path.isfile(p):
                return p

    # 2) Présent dans le PATH du processus ?
    trouve = shutil.which("ffmpeg")
    if trouve:
        return trouve

    # 3) Emplacements d'installation usuels
    candidats = []
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        # winget crée un lien dans \Links, et le binaire réel sous \Packages
        candidats.append(os.path.join(local, "Microsoft", "WinGet", "Links", exe))
        candidats += glob.glob(os.path.join(
            local, "Microsoft", "WinGet", "Packages",
            "Gyan.FFmpeg*", "**", "bin", exe), recursive=True)
    candidats += [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    for c in candidats:
        if c and os.path.isfile(c):
            return c
    return None


def trouver_ffprobe(chemin_ffmpeg=None):
    """Localise ffprobe : d'abord à côté du ffmpeg déjà trouvé (c'est le cas
    dans toutes les distributions officielles), sinon dans le PATH.
    ffprobe n'est pas indispensable : il sert uniquement à connaître la durée
    de l'audio pour alimenter la barre de progression du montage vidéo."""
    exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    if chemin_ffmpeg and os.path.isfile(chemin_ffmpeg):
        cand = os.path.join(os.path.dirname(chemin_ffmpeg), exe)
        if os.path.isfile(cand):
            return cand
    return shutil.which("ffprobe")


def options_subprocess():
    """Empêche l'ouverture d'une fenêtre console noire sous Windows lorsqu'on
    appelle ffmpeg / ffprobe depuis l'interface graphique."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def duree_media(chemin, ffprobe=None):
    """Durée d'un média en secondes. Utilise ffprobe si disponible, sinon
    soundfile pour les formats audio non compressés. Renvoie None si la durée
    ne peut pas être déterminée (la progression passera alors en mode
    indéterminé, sans empêcher le rendu)."""
    if ffprobe:
        try:
            res = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", chemin],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                **options_subprocess())
            valeur = float(res.stdout.decode("utf-8", errors="replace").strip())
            if valeur > 0:
                return valeur
        except Exception:
            pass
    try:
        sf = precharger_modules()["sf"]
        info = sf.info(chemin)
        return info.frames / float(info.samplerate)
    except Exception:
        return None


# =============================================================================
#  SECTION 4 — MOTEUR DE GÉNÉRATION (exécuté dans un thread de travail)
# =============================================================================

class Bus:
    """Canal de communication thread → interface via une file thread-safe."""
    def __init__(self, file):
        self.file = file

    def log(self, m):        self.file.put(("log", m))
    def status(self, m):     self.file.put(("status", m))
    def progress(self, i, n): self.file.put(("progress", i, n))
    def vram(self, g):       self.file.put(("vram", g))
    def termine(self, d):    self.file.put(("done", d))
    def erreur(self, m):     self.file.put(("error", m))
    def modele(self, ok):    self.file.put(("model", ok))

    # -- canaux dédiés au montage vidéo (pour ne pas interférer avec l'audio)
    def v_status(self, m):        self.file.put(("vstatus", m))
    def v_progress(self, pct):    self.file.put(("vprogress", pct))
    def v_termine(self, d):       self.file.put(("vdone", d))
    def v_erreur(self, m):        self.file.put(("verror", m))

    # -- canaux dédiés à la fusion audio
    def f_status(self, m):        self.file.put(("fstatus", m))
    def f_progress(self, pct):    self.file.put(("fprogress", pct))
    def f_chapitres(self, txt):   self.file.put(("fchapitres", txt))
    def f_termine(self, d):       self.file.put(("fdone", d))
    def f_erreur(self, m):        self.file.put(("ferror", m))


def charger_modele(config, bus, cache):
    """Charge (ou réutilise) le modèle Qwen3-TTS. cache est un dict persistant
    porté par l'application, qui évite de recharger inutilement le modèle."""
    mods = precharger_modules()           # déjà chargé sur le thread principal
    torch = mods["torch"]
    Qwen3TTSModel = mods["Qwen3TTSModel"]

    device = config["device"]
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Réutilisation si rien n'a changé
    if (cache.get("model") is not None
            and cache.get("id") == config["model_id"]
            and cache.get("device") == device):
        bus.log("Modèle déjà en mémoire — réutilisation (chargement instantané).")
        return cache["model"], device

    # Libération de l'ancien modèle si présent
    if cache.get("model") is not None:
        cache["model"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Affichage de l'environnement
    if torch.cuda.is_available() and device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        bus.log(f"GPU détecté : {gpu_name}  ({vram_total:.1f} Go VRAM)")
    elif not torch.cuda.is_available():
        bus.log("Aucun GPU détecté — exécution sur CPU (lent).")
    else:
        bus.log("CPU forcé par configuration.")

    bus.status(f"Chargement du modèle {config['model_id']}…")
    bus.log(f"Chargement du modèle {config['model_id']} (peut prendre un instant)…")

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    # Triple tentative reprenant la robustesse du script d'origine
    try:
        model = Qwen3TTSModel.from_pretrained(config["model_id"], dtype=dtype, device=device)
    except TypeError:
        try:
            model = Qwen3TTSModel.from_pretrained(config["model_id"], dtype=dtype, device_map=device)
        except TypeError:
            model = Qwen3TTSModel.from_pretrained(config["model_id"], dtype=dtype)
            if hasattr(model, "to"):
                model = model.to(device)
            else:
                bus.log("Impossible de déplacer le modèle sur le device — device par défaut utilisé.")

    if ACCELERATION:
        try:
            from acceleration_qwen import installer_acceleration
            bus.status("Préparation de l'accélération GPU…")
            libelle = installer_acceleration(
                model, torch, graphes=device.startswith("cuda"), journal=bus.log)
        except ImportError:
            libelle = "désactivée (acceleration_qwen.py absent du dossier du studio)"
        except Exception as e:  # noqa: BLE001
            libelle = f"désactivée ({e})"
        bus.log(f"Accélération du sous-modèle : {libelle}")

    cache.update(model=model, id=config["model_id"], device=device)
    bus.log("Moteur de clonage prêt.")
    if torch.cuda.is_available() and device.startswith("cuda"):
        vram_used = torch.cuda.memory_reserved(0) / 1e9
        bus.log(f"VRAM utilisée après chargement : {vram_used:.1f} Go")
    bus.modele(True)
    return model, device


# =============================================================================
#  SECTION 3 bis — TRANSCRIPTION (WHISPER) ET CONTRÔLE QUALITÉ
# =============================================================================
# Whisper passe par transformers, déjà installé pour Qwen3-TTS : aucun paquet
# supplémentaire. Le modèle (~1,6 Go) est téléchargé au premier usage.

WHISPER_MODELE = "openai/whisper-large-v3-turbo"
SEUIL_QC = 0.75          # conformité minimale entre le texte et ce qu'a entendu Whisper
QC_TENTATIVES = 2        # nouvelles générations au plus pour une phrase douteuse


def charger_whisper(cache, device, bus):
    """Charge (une seule fois) la reconnaissance vocale Whisper."""
    if cache.get("whisper") is not None:
        return cache["whisper"]
    torch = precharger_modules()["torch"]
    from transformers import pipeline
    bus.log(f"Chargement de Whisper ({WHISPER_MODELE}) — environ 1,6 Go "
            "téléchargés au premier usage…")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pipe = pipeline("automatic-speech-recognition", model=WHISPER_MODELE,
                    dtype=dtype, device=device)
    cache["whisper"] = pipe
    return pipe


def transcrire(pipe, audios, sr, langue, taille_lot=8):
    """Transcrit une liste de signaux (numpy, fréquence sr). Renvoie une
    liste de textes, dans le même ordre."""
    mods = precharger_modules()
    np, resample_poly = mods["np"], mods["resample_poly"]
    entrees = []
    for a in audios:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            d = gcd(16000, int(sr))
            a = resample_poly(a, 16000 // d, int(sr) // d).astype(np.float32)
        entrees.append(a)
    options = {"task": "transcribe"}
    if langue:
        options["language"] = str(langue).lower()
    textes = [""] * len(entrees)
    courts = [i for i, a in enumerate(entrees) if len(a) <= 30 * 16000]
    if courts:
        res = pipe([{"raw": entrees[i], "sampling_rate": 16000} for i in courts],
                   batch_size=max(1, taille_lot), generate_kwargs=options)
        for i, r in zip(courts, res):
            textes[i] = (r.get("text") or "").strip()
    for i, a in enumerate(entrees):              # au-delà de 30 s : mode long
        if len(a) > 30 * 16000:
            r = pipe({"raw": a, "sampling_rate": 16000},
                     return_timestamps=True, generate_kwargs=options)
            textes[i] = (r.get("text") or "").strip()
    return textes


def mots_pour_comparaison(texte):
    """Normalise un texte pour le comparer à une transcription : nombres en
    lettres, minuscules, sans ponctuation ni traits d'union."""
    t = nettoyer_texte_pour_audio(texte).lower()
    t = t.replace("’", "'").replace("-", " ")
    t = re.sub(r"[^\w\s']", " ", t)
    t = t.replace("'", "' ")
    return t.split()


def score_conformite(source, transcrit):
    """Ressemblance (0 à 1) entre le texte à lire et ce que Whisper a entendu."""
    import difflib
    a, b = mots_pour_comparaison(source), mots_pour_comparaison(transcrit)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def transcrire_fichier(chemin, langue, cache, device, bus):
    """Transcrit un fichier audio entier (voix de référence)."""
    sf = precharger_modules()["sf"]
    data, sr = sf.read(chemin, dtype="float32", always_2d=False)
    pipe = charger_whisper(cache, device, bus)
    duree = len(data) / sr
    return transcrire(pipe, [data], sr, langue)[0], duree


def worker_transcription(chemin, langue, device, bus, cache):
    """Thread : transcrit la voix de référence et renvoie le texte à l'interface."""
    try:
        torch = precharger_modules()["torch"]
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        bus.status("Transcription de la voix de référence…")
        texte, duree = transcrire_fichier(chemin, langue, cache, device, bus)
        bus.file.put(("transcription", texte, duree))
    except Exception as e:  # noqa: BLE001
        bus.file.put(("transcription_erreur", str(e)))


# =============================================================================
#  SECTION 3 ter — SOUS-TITRES, LIVRE AUDIO M4B, PROJET DE CORRECTION
# =============================================================================

def _horodate_srt(secondes):
    ms = int(round(max(0.0, secondes) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _couper_lignes(texte, largeur=42):
    """Coupe un sous-titre en deux lignes équilibrées, sur un espace."""
    if len(texte) <= largeur:
        return texte
    milieu = len(texte) // 2
    espaces = [i for i, c in enumerate(texte) if c == " "]
    if not espaces:
        return texte
    i = min(espaces, key=lambda x: abs(x - milieu))
    return texte[:i] + "\n" + texte[i + 1:]


def ecrire_srt(segments, chemin, max_car=84):
    """segments : [(debut_s, fin_s, texte)]. Un chunk trop long est réparti en
    plusieurs sous-titres, au prorata du nombre de caractères."""
    blocs, n = [], 0
    for debut, fin, texte in segments:
        parties, courant = [], ""
        for mot in texte.split():
            if courant and len(courant) + 1 + len(mot) > max_car:
                parties.append(courant)
                courant = mot
            else:
                courant = (courant + " " + mot).strip()
        if courant:
            parties.append(courant)
        total = sum(len(p) for p in parties) or 1
        t = debut
        for p in parties:
            d = (fin - debut) * len(p) / total
            n += 1
            blocs.append(f"{n}\n{_horodate_srt(t)} --> {_horodate_srt(t + d)}\n"
                         f"{_couper_lignes(p)}\n")
            t += d
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(blocs))
    return chemin


def creer_m4b(ffmpeg, source, chapitres, duree_totale, sortie, titre, auteur,
              couverture, bus, debit="96k"):
    """Livre audio M4B : audio AAC, chapitres navigables, titre, auteur et
    couverture. chapitres : [{"debut", "libelle"}]."""
    meta = os.path.splitext(sortie)[0] + ".ffmeta.txt"
    lignes = [";FFMETADATA1"]
    for cle, valeur in (("title", titre), ("album", titre), ("artist", auteur),
                        ("album_artist", auteur), ("genre", "Audiobook")):
        if valeur:
            lignes.append(f"{cle}={_echapper_ffmetadata(valeur)}")
    for i, ch in enumerate(chapitres):
        fin = chapitres[i + 1]["debut"] if i + 1 < len(chapitres) else duree_totale
        lignes += ["[CHAPTER]", "TIMEBASE=1/1000",
                   f"START={int(ch['debut'] * 1000)}",
                   f"END={int(max(fin, ch['debut'] + 0.001) * 1000)}",
                   f"title={_echapper_ffmetadata(ch['libelle'])}"]
    with open(meta, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-i", source, "-i", meta]
    avec_image = bool(couverture) and os.path.isfile(couverture)
    if avec_image:
        cmd += ["-i", couverture]
    cmd += ["-map", "0:a"]
    if avec_image:
        cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
    cmd += ["-map_metadata", "1", "-map_chapters", "1",
            "-c:a", "aac", "-b:a", debit, "-ac", "1",
            "-movflags", "+faststart", "-f", "mp4", sortie]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, **options_subprocess())
        bus.log(f"   ✓ Livre audio M4B ({len(chapitres)} chapitres) : {os.path.abspath(sortie)}")
        return True
    except subprocess.CalledProcessError as e:
        detail = e.stderr.decode("utf-8", errors="replace").strip()[-300:] if e.stderr else ""
        bus.log(f"   ❌ Création du M4B impossible : {detail}")
    except Exception as e:  # noqa: BLE001
        bus.log(f"   ❌ Création du M4B impossible : {e}")
    finally:
        try:
            os.unlink(meta)
        except Exception:
            pass
    return False


def dossier_projet(chemin_wav):
    """Dossier où sont conservés les morceaux d'un rendu (pour correction)."""
    return os.path.splitext(chemin_wav)[0] + "_projet"


def sauver_projet(projet, audios, sr):
    """Enregistre chaque morceau (WAV 32 bits flottants) et le fichier
    projet.json qui décrit le rendu."""
    sf = precharger_modules()["sf"]
    dossier = dossier_projet(projet["sortie"]["wav"])
    os.makedirs(os.path.join(dossier, "morceaux"), exist_ok=True)
    for k, c in enumerate(projet["chunks"]):
        if k < len(audios) and audios[k] is not None:
            nom = f"{k + 1:04d}.wav"
            sf.write(os.path.join(dossier, "morceaux", nom), audios[k], sr, subtype="FLOAT")
            c["fichier"] = nom
    projet["sr"] = sr
    with open(os.path.join(dossier, "projet.json"), "w", encoding="utf-8") as f:
        json.dump(projet, f, ensure_ascii=False, indent=1)
    return dossier


def charger_projet(dossier):
    """Relit projet.json et les morceaux d'un rendu précédent."""
    sf = precharger_modules()["sf"]
    with open(os.path.join(dossier, "projet.json"), encoding="utf-8") as f:
        projet = json.load(f)
    audios = []
    for c in projet["chunks"]:
        chemin = os.path.join(dossier, "morceaux", c.get("fichier") or "")
        if c.get("fichier") and os.path.isfile(chemin):
            data, _ = sf.read(chemin, dtype="float32")
            audios.append(data)
        else:
            audios.append(None)
    return projet, audios


def assembler_et_exporter(projet, audios, sr, limite, bus):
    """Assemble les morceaux dans l'ordre (avec les silences) et produit
    toutes les sorties demandées : WAV, version rééchantillonnée, MP3,
    sous-titres SRT, livre audio M4B et chapitrage. Renvoie un dict
    {fichiers, principal, duree} ou None si rien n'a été généré."""
    mods = precharger_modules()
    np, sf, resample_poly = mods["np"], mods["sf"], mods["resample_poly"]
    r = projet["reglages"]
    chunks = projet["chunks"]

    pieces, segments, debuts = [], [], {}
    curseur = 0
    for k in range(min(limite, len(chunks))):
        a = audios[k]
        if a is None:
            continue
        c = chunks[k]
        a = normaliser_peak(a, r["target_peak"])
        debuts.setdefault(c["chapitre"], curseur / sr)
        segments.append((curseur / sr, (curseur + len(a)) / sr, c["texte"]))
        pieces.append(a)
        curseur += len(a)
        dur = duree_silence_apres(c["texte"], c["fin_para"],
                                  r["sil_virgule"], r["sil_point"], r["sil_para"])
        silence = np.zeros(int(dur * sr), dtype=np.float32)
        pieces.append(silence)
        curseur += len(silence)
    if not pieces:
        return None

    audio_complet = np.concatenate(pieces).astype(np.float32)
    chemin_wav = projet["sortie"]["wav"]
    os.makedirs(os.path.dirname(os.path.abspath(chemin_wav)), exist_ok=True)
    sf.write(chemin_wav, audio_complet, sr)
    duree = len(audio_complet) / sr
    base = os.path.splitext(chemin_wav)[0]
    bus.log(f"🎭 Fichier natif ({sr} Hz) : {os.path.abspath(chemin_wav)}")
    bus.log(f"   Durée : {int(duree // 60)} min {int(duree % 60)} sec")
    fichiers = [chemin_wav]

    # --- Version rééchantillonnée ---
    chemin_resample = None
    freq = r.get("freq_supp")
    if freq and int(freq) != sr:
        fs = int(freq)
        d = gcd(fs, sr)
        bus.log(f"   Rééchantillonnage {sr} → {fs} Hz…")
        audio_rs = resample_poly(audio_complet, up=fs // d, down=sr // d).astype(np.float32)
        chemin_resample = f"{base}-{fs}.wav"
        sf.write(chemin_resample, audio_rs, fs)
        bus.log(f"   ✓ Version {fs} Hz : {os.path.abspath(chemin_resample)}")
        fichiers.append(chemin_resample)
        audio_rs = None

    ffmpeg = None
    if r.get("export_mp3") or r.get("livre"):
        ffmpeg = trouver_ffmpeg(r.get("ffmpeg_path"))
        if ffmpeg is None:
            bus.log("   ⚠️  ffmpeg introuvable — MP3 et livre audio ignorés. "
                    "Indique son chemin dans l'onglet Export.")

    # --- MP3 ---
    if r.get("export_mp3") and ffmpeg:
        src = r.get("mp3_source", "auto")
        if src == "resample" and chemin_resample:
            source_mp3 = chemin_resample
        elif src == "natif":
            source_mp3 = chemin_wav
        else:
            source_mp3 = chemin_resample or chemin_wav
        nom_mp3 = os.path.splitext(source_mp3)[0] + ".mp3"
        cmd = [ffmpeg, "-y", "-i", source_mp3, "-c:a", "libmp3lame",
               "-b:a", r.get("mp3_bitrate", "192k"),
               "-ac", "1" if r.get("mp3_mono", True) else "2", nom_mp3]
        bus.log(f"   Encodage MP3 ({r.get('mp3_bitrate', '192k')})…")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, **options_subprocess())
            bus.log(f"   ✓ MP3 : {os.path.abspath(nom_mp3)}")
            fichiers.append(nom_mp3)
        except subprocess.CalledProcessError as e:
            bus.log(f"   ❌ Erreur ffmpeg (code {e.returncode}).")
            if e.stderr:
                bus.log(e.stderr.decode("utf-8", errors="replace")[-400:])

    # --- Sous-titres ---
    if r.get("srt"):
        chemin_srt = ecrire_srt(segments, base + ".srt")
        bus.log(f"   ✓ Sous-titres : {os.path.abspath(chemin_srt)}")
        fichiers.append(chemin_srt)

    # --- Chapitres et livre audio ---
    titres = projet.get("titres") or []
    if r.get("livre") and any(titres):
        chapitres = [{"debut": debuts[i], "libelle": titres[i] or f"Chapitre {i + 1}"}
                     for i in sorted(debuts)]
        chapitres[0]["debut"] = 0.0
        chemin_chap = base + "-chapitres.txt"
        with open(chemin_chap, "w", encoding="utf-8") as f:
            f.write(formater_chapitres(chapitres, duree, "youtube") + "\n")
        bus.log(f"   ✓ Chapitrage ({len(chapitres)} chapitres) : {os.path.abspath(chemin_chap)}")
        fichiers.append(chemin_chap)
        if ffmpeg:
            chemin_m4b = base + ".m4b"
            if creer_m4b(ffmpeg, chemin_wav, chapitres, duree, chemin_m4b,
                         r.get("livre_titre", ""), r.get("livre_auteur", ""),
                         r.get("livre_couverture", ""), bus):
                fichiers.append(chemin_m4b)
    elif r.get("livre"):
        bus.log("   ⚠ Aucun titre trouvé dans le document : pas de chapitres. "
                "Utilise les styles « Titre 1 » de Word ou des « # » en Markdown.")

    return {"fichiers": fichiers, "principal": chemin_wav, "duree": duree}


def fabriquer_generateur(model, prompts, voix_par_chunk, chunks, langue, bus):
    """Renvoie generer(indices) -> (wavs, sr) : génère d'un coup les chunks
    demandés, chacun avec sa voix, et plafonne la durée contre les boucles."""
    def plafond_trames(textes):
        # Le modèle produit 12,5 trames par seconde, soit ~0,6 trame par
        # caractère en français. Plafond à ~4x cette valeur : une phrase
        # qui « boucle » (le modèle ne s'arrête pas) ne peut plus retenir
        # tout son lot jusqu'à la limite d'origine de 2 048 trames (2 min 44).
        return int(40 + 2.5 * max(len(t) for t in textes))

    def generer(indices):
        textes = [chunks[k][0] for k in indices]
        plafond = plafond_trames(textes)
        # Le modèle Base ignore toute consigne de ton : seule la voix de
        # référence compte. On ne transmet donc pas `instruct`.
        wavs, sr = model.generate_voice_clone(
            text=textes,
            language=langue,
            voice_clone_prompt=[prompts[voix_par_chunk[k]] for k in indices],
            max_new_tokens=plafond,
        )
        limite = 0.95 * plafond / 12.5 * sr
        for k, t, w in zip(indices, textes, wavs):
            if len(w) >= limite:
                bus.log(f"   ⚠ Phrase {k + 1} coupée à la durée maximale (le modèle a "
                        f"probablement bouclé) : « {t[:50]}… » — à vérifier à l'écoute.")
        return wavs, sr
    return generer


def preparer_voix(model, config, noms, bus, cache, device):
    """Empreintes vocales : voix principale (clé None) et voix des balises
    [Nom] trouvées dans le dossier des voix. Une voix sans transcription
    (Nom.txt) est transcrite par Whisper, et la transcription enregistrée.
    Renvoie (prompts, sources) ; sources décrit chaque voix pour le projet."""
    sf = precharger_modules()["sf"]
    ref, ref_text = config["ref_audio"], (config.get("ref_text") or "").strip()
    sources = {None: {"audio": ref, "transcription": ref_text}}
    manquantes = []
    for nom in noms:
        trouve = trouver_voix(config.get("dossier_voix"), nom)
        if not trouve:
            manquantes.append(nom)
            continue
        audio, chemin_txt = trouve
        transcription = ""
        if chemin_txt:
            with open(chemin_txt, encoding="utf-8", errors="replace") as f:
                transcription = f.read().strip()
        if not transcription:
            try:
                transcription, _ = transcrire_fichier(audio, config["language"],
                                                      cache, device, bus)
                with open(os.path.splitext(audio)[0] + ".txt", "w", encoding="utf-8") as f:
                    f.write(transcription)
                bus.log(f"   🎧 Voix « {nom} » transcrite automatiquement "
                        f"(enregistrée dans {os.path.basename(os.path.splitext(audio)[0])}.txt).")
            except Exception as e:  # noqa: BLE001
                bus.log(f"   ⚠ Transcription automatique de « {nom} » impossible ({e}) : "
                        "empreinte seule.")
        sources[nom] = {"audio": audio, "transcription": transcription}
    if manquantes:
        bus.log("⚠ Voix introuvables dans le dossier des voix : "
                + ", ".join(manquantes) + " — la voix principale les remplace.")

    prompts = {}
    for nom, s in sources.items():
        try:
            prompts[nom] = model.create_voice_clone_prompt(
                ref_audio=s["audio"], ref_text=s["transcription"] or None,
                x_vector_only_mode=not s["transcription"])[0]
        except Exception as e:  # noqa: BLE001
            if nom is None:
                raise
            bus.log(f"⚠ Voix « {nom} » inutilisable ({e}) — voix principale à la place.")
    sources = {k: v for k, v in sources.items() if k in prompts}
    try:
        duree = sf.info(ref).duration
        if duree > 20:
            bus.log(f"💡 Voix de référence de {duree:.0f} s : un extrait de 10 à 15 s "
                    "réduirait la mémoire utilisée et permettrait des lots plus grands.")
    except Exception:  # noqa: BLE001
        pass
    return prompts, sources


def controle_qualite(generer, resultats, chunks, sr, langue, lot, cache, device,
                     bus, stop_event):
    """Fait écouter chaque phrase par Whisper, compare au texte et régénère
    les phrases douteuses (au plus QC_TENTATIVES fois), en gardant la
    meilleure version. Renvoie {indice: score}."""
    try:
        pipe = charger_whisper(cache, device, bus)
    except Exception as e:  # noqa: BLE001
        bus.log(f"⚠ Contrôle qualité impossible : Whisper n'a pas pu être chargé ({e}).")
        return {}
    bus.status("Contrôle qualité (Whisper)…")
    indices = [k for k in range(len(resultats)) if resultats[k] is not None]
    t = time.time()
    transcrits = transcrire(pipe, [resultats[k] for k in indices], sr, langue,
                            taille_lot=min(lot, 16))
    scores = {k: score_conformite(chunks[k][0], tr) for k, tr in zip(indices, transcrits)}
    douteux = [k for k in indices if scores[k] < SEUIL_QC]
    bus.log(f"🔎 Contrôle qualité : {len(indices)} phrases écoutées en "
            f"{time.time() - t:.0f} s, {len(douteux)} à reprendre.")

    for essai in range(1, QC_TENTATIVES + 1):
        if not douteux or stop_event.is_set():
            break
        bus.log(f"   Nouvelle génération de {len(douteux)} phrase(s) (essai {essai}/{QC_TENTATIVES})…")
        nouveaux = {}
        for b0 in range(0, len(douteux), lot):
            groupe = douteux[b0:b0 + lot]
            try:
                wavs, _ = generer(groupe)
                nouveaux.update(zip(groupe, wavs))
            except Exception as e:  # noqa: BLE001
                bus.log(f"   ❌ Régénération impossible : {e}")
        if not nouveaux:
            break
        cles = list(nouveaux)
        trs = transcrire(pipe, [nouveaux[k] for k in cles], sr, langue, taille_lot=min(lot, 16))
        for k, tr in zip(cles, trs):
            s = score_conformite(chunks[k][0], tr)
            if s > scores[k]:
                resultats[k], scores[k] = nouveaux[k], s
        douteux = [k for k in douteux if scores[k] < SEUIL_QC]

    if douteux:
        liste = ", ".join(str(k + 1) for k in douteux[:25]) + ("…" if len(douteux) > 25 else "")
        bus.log(f"   ⚠ {len(douteux)} phrase(s) encore douteuse(s) : n° {liste}. "
                "À écouter, ou à reprendre avec « Corriger des phrases ».")
    else:
        bus.log("   ✓ Toutes les phrases correspondent au texte.")
    return scores


def worker_generation(config, bus, stop_event, cache):
    """Fonction exécutée dans le thread de travail : lecture, découpage,
    génération par lots, contrôle qualité, assemblage et exports."""
    mods = precharger_modules()           # renvoie le cache (import déjà fait)
    np = mods["np"]

    t0 = time.time()

    try:
        # --- Authentification Hugging Face (optionnelle) ---
        token = config.get("hf_token", "").strip()
        if token:
            from huggingface_hub import login
            login(token=token)
            bus.log("Authentification Hugging Face effectuée.")

        # --- Vérification des fichiers ---
        ref = config["ref_audio"]
        txt = config["text_file"]
        if not os.path.exists(ref):
            bus.erreur(f"Fichier de référence introuvable : {ref}")
            return
        if not os.path.exists(txt):
            bus.erreur(f"Fichier texte introuvable : {txt}")
            return

        # --- Lecture + découpage ---
        bus.status("Lecture et préparation du texte…")
        bus.log(f"Lecture du fichier : {os.path.basename(txt)}")
        plan = construire_plan(lire_paragraphes(txt), config["max_chars"],
                               par_chapitres=bool(config.get("livre")))
        chunks_finaux, stats = plan["chunks"], plan["stats"]
        bus.log(f"Texte découpé en {stats['nb_chunks']} chunks "
                f"(min {stats['min']} / max {stats['max']} / moy {stats['moyenne']} car.)")
        bus.log(f"Fins de paragraphe détectées : {stats['nb_fin_para']}")
        if stats["voix"]:
            bus.log(f"Voix balisées : {', '.join(stats['voix'])}")
        if config.get("livre"):
            bus.log(f"Chapitres détectés : {stats['nb_chapitres']}")

        if not chunks_finaux:
            bus.erreur("Aucun contenu exploitable après nettoyage du texte.")
            return

        # --- Chargement du modèle ---
        model, device = charger_modele(config, bus, cache)

        torch = mods["torch"]
        sur_gpu = torch.cuda.is_available() and device.startswith("cuda")

        # --- Empreintes vocales : calculées UNE SEULE FOIS par voix ---
        bus.status("Analyse des voix…")
        try:
            prompts, sources = preparer_voix(model, config, stats["voix"], bus, cache, device)
        except Exception as e:  # noqa: BLE001
            bus.erreur(f"Analyse de la voix de référence impossible : {e}")
            return
        if not sources[None]["transcription"]:
            bus.log("⚠ Transcription de référence vide — clonage sur la seule "
                    "empreinte du locuteur (moins fidèle).")
        voix_par_chunk = [v if v in prompts else None for v in plan["voix"]]
        generer = fabriquer_generateur(model, prompts, voix_par_chunk, chunks_finaux,
                                       config["language"], bus)

        # --- Boucle de génération PAR LOTS ---
        # La génération est autorégressive : à une phrase par appel, le GPU
        # attend la mémoire bien plus qu'il ne calcule. Plusieurs phrases par
        # appel coûtent presque le même temps qu'une seule.
        # Dans chaque fenêtre de phrases consécutives, on regroupe celles de
        # longueur voisine : un lot dure aussi longtemps que sa plus longue.
        lot = max(1, int(config.get("taille_lot", TAILLE_LOT) or 1))
        n = len(chunks_finaux)
        resultats = [None] * n          # audio de chaque chunk (None = échec)
        fait = [False] * n              # chunk traité (réussi ou non)
        sr_final = 24000
        echecs = 0
        traites = 0
        vram_totale = (torch.cuda.get_device_properties(0).total_memory
                       if sur_gpu else 0)
        alerte_vram = False
        t_gen = time.time()
        bus.status("Génération en cours…")
        bus.log(f"Début du clonage vocal — {n} morceaux sur {device}, "
                f"par lots de {lot}.")

        lot_initial = lot
        f0 = 0
        while f0 < n:
            if stop_event.is_set():
                break
            fenetre = lot * 2
            indices = sorted(range(f0, min(n, f0 + fenetre)),
                             key=lambda k: len(chunks_finaux[k][0]))
            f0 += len(indices)
            t_fen = time.time()

            b0 = 0
            while b0 < len(indices):
                if stop_event.is_set():
                    break
                groupe = indices[b0:b0 + lot]
                b0 += len(groupe)

                for k in sorted(groupe):
                    chunk, est_fin_para = chunks_finaux[k]
                    apercu = chunk[:55] + "…" if len(chunk) > 55 else chunk
                    marque = " ¶" if est_fin_para else ""
                    qui = f" [{voix_par_chunk[k]}]" if voix_par_chunk[k] else ""
                    bus.log(f"[{k + 1:>4}/{n}] ({len(chunk):>3} car.){marque}{qui} : {apercu}")

                try:
                    wavs, sr = generer(groupe)
                    for k, w in zip(groupe, wavs):
                        resultats[k] = w
                    sr_final = sr
                except Exception as e:
                    if len(groupe) == 1:
                        echecs += 1
                        bus.log(f"   ❌ Erreur ignorée sur ce chunk : {e}")
                    else:
                        # Typiquement un manque de VRAM : on reprend une à une.
                        bus.log(f"   ⚠ Lot en échec ({e}) — reprise phrase par phrase.")
                        if sur_gpu:
                            torch.cuda.empty_cache()
                        for k in groupe:
                            try:
                                wavs, sr = generer([k])
                                resultats[k] = wavs[0]
                                sr_final = sr
                            except Exception as e2:
                                echecs += 1
                                bus.log(f"   ❌ Erreur ignorée sur le chunk {k + 1} : {e2}")

                for k in groupe:
                    fait[k] = True
                traites += len(groupe)
                bus.progress(traites, n)

                # Débordement de VRAM : Windows bascule alors sur la mémoire
                # système et tout ralentit (jusqu'à 10x). On réduit le lot
                # aussitôt, pour la suite du rendu.
                if sur_gpu and torch.cuda.memory_reserved(0) > 0.95 * vram_totale:
                    if lot > 1:
                        lot = max(1, lot // 2)
                        torch.cuda.empty_cache()
                        bus.log(f"   ⚠ VRAM pleine (débordement sur la mémoire système) — "
                                f"lot réduit à {lot} pour la suite.")
                    elif not alerte_vram:
                        alerte_vram = True
                        bus.log("   ⚠ VRAM pleine même phrase par phrase : "
                                "raccourcis la voix de référence ou la taille des chunks.")

            nb_fen = sum(1 for k in indices if fait[k])
            if nb_fen:
                bus.log(f"         ⏱ {nb_fen} phrases en {time.time() - t_fen:.1f} s")
            if sur_gpu:
                bus.vram(torch.cuda.memory_reserved(0) / 1e9)

        if stop_event.is_set():
            bus.log("⏹ Arrêt demandé par l'utilisateur.")
        if lot < lot_initial:
            bus.log(f"💡 Le lot a dû être réduit de {lot_initial} à {lot} : "
                    f"mets TAILLE_LOT = {lot} en tête de qwen_tts_studio.py.")
        bus.progress(traites, n)
        bus.log(f"Génération : {time.time() - t_gen:.0f} s pour {traites} morceaux.")

        # --- Contrôle qualité (optionnel) ---
        scores = {}
        if config.get("qc") and not stop_event.is_set():
            scores = controle_qualite(generer, resultats, chunks_finaux, sr_final,
                                      config["language"], lot, cache, device,
                                      bus, stop_event)

        # --- Remise dans l'ordre du texte ---
        # Après un arrêt, on ne garde que la partie continue depuis le début :
        # pas de trou au milieu du rendu partiel.
        limite = n
        arret = stop_event.is_set()
        if arret:
            limite = next((k for k in range(n) if not fait[k]), n)
        if sur_gpu:
            torch.cuda.empty_cache()
        gc.collect()

        if not any(resultats[k] is not None for k in range(limite)):
            bus.erreur("Aucun audio n'a pu être généré.")
            return

        # --- Nom des fichiers ---
        base = config["output_base"].strip() or "rendu"
        suffixe = "-partiel" if arret else ""
        if config["horodater"]:
            nom = f"{base}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{suffixe}.wav"
        else:
            nom = f"{base}{suffixe}.wav"
        out_dir = config["output_dir"].strip() or os.path.dirname(txt) or "."
        os.makedirs(out_dir, exist_ok=True)

        projet = {
            "version": 1,
            "texte_source": txt,
            "langue": config["language"],
            "voix": {("" if k is None else k): v for k, v in sources.items()},
            "reglages": {cle: config.get(cle) for cle in (
                "sil_virgule", "sil_point", "sil_para", "target_peak", "freq_supp",
                "export_mp3", "mp3_bitrate", "mp3_mono", "mp3_source", "ffmpeg_path",
                "srt", "livre", "livre_titre", "livre_auteur", "livre_couverture")},
            "sortie": {"wav": os.path.join(out_dir, nom)},
            "titres": plan["titres"],
            "limite": limite,
            "chunks": [{"texte": c, "fin_para": fp, "voix": voix_par_chunk[k] or "",
                        "chapitre": plan["chapitre"][k],
                        "score": round(scores[k], 3) if k in scores else None}
                       for k, (c, fp) in enumerate(chunks_finaux)],
        }
        if config.get("livre") and not projet["reglages"].get("livre_titre"):
            projet["reglages"]["livre_titre"] = base

        # --- Assemblage et exports ---
        bus.status("Assemblage final…")
        bus.log(f"Assemblage… ({echecs} échec(s) sur {n} chunks)")
        res = assembler_et_exporter(projet, resultats, sr_final, limite, bus)
        if res is None:
            bus.erreur("Aucun audio n'a pu être généré.")
            return

        dossier = None
        if config.get("garder_morceaux", True):
            try:
                dossier = sauver_projet(projet, resultats, sr_final)
                bus.log(f"🧩 Morceaux conservés pour correction : {dossier}")
            except Exception as e:  # noqa: BLE001
                bus.log(f"   ⚠ Morceaux non conservés : {e}")

        bus.termine({
            "fichiers": res["fichiers"],
            "principal": res["principal"],
            "dossier": os.path.abspath(out_dir),
            "duree_audio": res["duree"],
            "duree_calcul": time.time() - t0,
            "echecs": echecs,
            "total": n,
            "partiel": arret,
            "projet": dossier,
        })

    except Exception as e:
        import traceback
        bus.log(traceback.format_exc())
        bus.erreur(str(e))


def worker_correction(dossier, indices, config, bus, stop_event, cache):
    """Régénère quelques phrases d'un rendu conservé, puis refait
    l'assemblage et toutes les sorties (WAV, MP3, SRT, M4B…)."""
    t0 = time.time()
    try:
        projet, audios = charger_projet(dossier)
        chunks = [(c["texte"], c["fin_para"]) for c in projet["chunks"]]
        sr = int(projet.get("sr") or 24000)
        model, device = charger_modele(config, bus, cache)

        voix_projet = {(k or None): v for k, v in projet["voix"].items()}
        utiles = {projet["chunks"][k]["voix"] or None for k in indices}
        prompts = {}
        for nom in utiles | {None}:
            s = voix_projet.get(nom) or voix_projet[None]
            prompts[nom] = model.create_voice_clone_prompt(
                ref_audio=s["audio"], ref_text=s["transcription"] or None,
                x_vector_only_mode=not s["transcription"])[0]
        voix_par_chunk = [(c["voix"] or None) if (c["voix"] or None) in prompts else None
                          for c in projet["chunks"]]
        generer = fabriquer_generateur(model, prompts, voix_par_chunk, chunks,
                                       projet["langue"], bus)

        lot = max(1, int(config.get("taille_lot", TAILLE_LOT) or 1))
        bus.status("Correction en cours…")
        bus.log(f"✎ Nouvelle génération de {len(indices)} phrase(s)…")
        faits, echecs = 0, 0
        for b0 in range(0, len(indices), lot):
            if stop_event.is_set():
                break
            groupe = indices[b0:b0 + lot]
            for k in groupe:
                bus.log(f"[{k + 1:>4}] {chunks[k][0][:70]}")
            try:
                wavs, sr = generer(groupe)
                for k, w in zip(groupe, wavs):
                    audios[k] = w
                    projet["chunks"][k]["score"] = None
            except Exception as e:  # noqa: BLE001
                echecs += len(groupe)
                bus.log(f"   ❌ {e}")
            faits += len(groupe)
            bus.progress(faits, len(indices))

        bus.status("Assemblage final…")
        res = assembler_et_exporter(projet, audios, sr,
                                    projet.get("limite", len(chunks)), bus)
        if res is None:
            bus.erreur("Aucun audio à assembler.")
            return
        sauver_projet(projet, audios, sr)
        bus.termine({
            "fichiers": res["fichiers"],
            "principal": res["principal"],
            "dossier": os.path.dirname(os.path.abspath(res["principal"])),
            "duree_audio": res["duree"],
            "duree_calcul": time.time() - t0,
            "echecs": echecs,
            "total": len(indices),
            "partiel": stop_event.is_set(),
            "projet": dossier,
        })
    except Exception as e:
        import traceback
        bus.log(traceback.format_exc())
        bus.erreur(str(e))


# =============================================================================
#  SECTION 4 bis — MONTAGE VIDÉO (image fixe + piste audio, via ffmpeg)
# =============================================================================

def construire_filtre_video(resolution, couleur_fond):
    """Chaîne de filtres ffmpeg appliquée à l'image fixe.

    • « source » : on conserve la taille de l'image d'origine, en forçant des
      dimensions paires (libx264 en yuv420p l'exige).
    • sinon : l'image est redimensionnée SANS déformation dans le cadre choisi,
      puis centrée sur un fond uni (letterbox / pillarbox)."""
    if not resolution or resolution == "source":
        return "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
    largeur, hauteur = resolution.split("x")
    largeur, hauteur = int(largeur), int(hauteur)
    couleur = (couleur_fond or "black").strip() or "black"
    return (
        f"scale={largeur}:{hauteur}:force_original_aspect_ratio=decrease,"
        f"pad={largeur}:{hauteur}:(ow-iw)/2:(oh-ih)/2:color={couleur},"
        f"setsar=1"
    )


def construire_commande_video(config, ffmpeg):
    """Assemble la ligne de commande ffmpeg du montage."""
    fps = max(1, int(config["fps"]))
    sortie = config["sortie"]
    ext = os.path.splitext(sortie)[1].lower()

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        # Entrée 0 : l'image, bouclée à l'infini (c'est -shortest qui coupera)
        "-loop", "1", "-framerate", str(fps), "-i", config["image"],
        # Entrée 1 : la piste audio
        "-i", config["audio"],
        # Sélection explicite des flux : évite que ffmpeg ne choisisse la
        # pochette intégrée d'un MP3 à la place de l'image sélectionnée.
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", construire_filtre_video(config["resolution"], config["couleur_fond"]),
        "-c:v", "libx264",
        "-preset", config["preset"],
        "-tune", "stillimage",
        "-crf", str(int(config["crf"])),
        "-r", str(fps),
        "-g", str(max(2, fps * 2)),
        "-pix_fmt", "yuv420p",
    ]

    if config["audio_copie"]:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", config["audio_bitrate"]]

    # S'arrête à la fin de la piste audio (l'image, elle, boucle sans fin)
    cmd += ["-shortest"]

    if ext in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]   # lecture en streaming immédiate

    # Progression machine sur la sortie standard
    cmd += ["-progress", "pipe:1", sortie]
    return cmd


def _secondes_depuis_horodate(valeur):
    """Convertit « HH:MM:SS.micro » (format de progression ffmpeg) en secondes.
    Renvoie None si la valeur est absente ou non exploitable."""
    try:
        heures, minutes, secondes = valeur.split(":")
        return int(heures) * 3600 + int(minutes) * 60 + float(secondes)
    except (ValueError, AttributeError):
        return None


def worker_video(config, bus, stop_event):
    """Thread de travail du montage vidéo : vérifications, appel de ffmpeg et
    suivi de progression en temps réel."""
    import tempfile
    t0 = time.time()
    proc = None
    fichier_err = None

    try:
        image = config["image"]
        audio = config["audio"]
        sortie = config["sortie"]

        if not os.path.exists(image):
            bus.v_erreur(f"Image introuvable : {image}")
            return
        if not os.path.exists(audio):
            bus.v_erreur(f"Fichier audio introuvable : {audio}")
            return
        if not sortie:
            bus.v_erreur("Aucun fichier vidéo de sortie n'a été indiqué.")
            return

        ffmpeg = trouver_ffmpeg(config.get("ffmpeg_path"))
        if ffmpeg is None:
            bus.v_erreur(
                "ffmpeg est introuvable : le montage vidéo en dépend entièrement.\n\n"
                "Installe-le (winget install --id=Gyan.FFmpeg -e) ou indique son "
                "chemin dans l'onglet « Export MP3 »."
            )
            return

        dossier = os.path.dirname(os.path.abspath(sortie))
        os.makedirs(dossier, exist_ok=True)

        ffprobe = trouver_ffprobe(ffmpeg)
        duree = duree_media(audio, ffprobe)

        bus.log("═" * 50)
        bus.log("🎬 Montage vidéo")
        bus.log(f"   Image  : {os.path.basename(image)}")
        bus.log(f"   Audio  : {os.path.basename(audio)}")
        bus.log(f"   Sortie : {os.path.abspath(sortie)}")
        if duree:
            bus.log(f"   Durée  : {int(duree // 60)} min {int(duree % 60)} sec")
        else:
            bus.log("   Durée  : indéterminée (progression approximative)")
        bus.log(f"   Format : {config['resolution']} • {config['fps']} img/s • "
                f"CRF {config['crf']} • preset {config['preset']}")

        cmd = construire_commande_video(config, ffmpeg)
        bus.log("   Commande : " + " ".join(
            f'"{a}"' if " " in a else a for a in cmd))
        bus.v_status("Encodage de la vidéo…")

        # stderr redirigé vers un fichier temporaire : on lit stdout en continu
        # pour la progression, sans risque d'interblocage si ffmpeg bavarde.
        fichier_err = tempfile.NamedTemporaryFile(
            prefix="qwen_video_", suffix=".log", delete=False)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=fichier_err,
            bufsize=1, text=True, encoding="utf-8", errors="replace",
            **options_subprocess())

        annule = False
        for ligne in proc.stdout:
            if stop_event.is_set() and not annule:
                annule = True
                bus.log("⏹ Arrêt demandé — interruption de ffmpeg.")
                try:
                    proc.terminate()
                except Exception:
                    pass
                break
            ligne = ligne.strip()
            # On lit « out_time=HH:MM:SS.micro », présent dans toutes les
            # versions de ffmpeg (out_time_ms est ambigu selon les versions).
            if ligne.startswith("out_time="):
                horodate = ligne.split("=", 1)[1].strip()
                secondes = _secondes_depuis_horodate(horodate)
                if secondes is None:
                    continue
                if duree:
                    bus.v_progress(min(100.0, secondes / duree * 100.0))
                    bus.v_status(f"Encodage… {int(secondes // 60)}m "
                                 f"{int(secondes % 60):02d}s / "
                                 f"{int(duree // 60)}m {int(duree % 60):02d}s")
                else:
                    bus.v_status(f"Encodage… {int(secondes // 60)}m "
                                 f"{int(secondes % 60):02d}s produites")

        proc.wait()
        try:
            fichier_err.close()
        except Exception:
            pass

        if annule or stop_event.is_set():
            bus.v_erreur("Montage vidéo annulé. Le fichier de sortie est "
                         "probablement incomplet.")
            return

        if proc.returncode != 0:
            detail = ""
            try:
                with open(fichier_err.name, encoding="utf-8", errors="replace") as fp:
                    detail = fp.read().strip()[-800:]
            except Exception:
                pass
            bus.log(detail or "(aucun détail fourni par ffmpeg)")
            bus.v_erreur(f"ffmpeg a échoué (code {proc.returncode}).\n\n"
                         f"{detail[-400:] if detail else 'Voir le journal.'}")
            return

        bus.v_progress(100.0)
        taille = os.path.getsize(sortie) / (1024 * 1024) if os.path.exists(sortie) else 0.0
        elapsed = time.time() - t0
        bus.log(f"   ✓ Vidéo : {os.path.abspath(sortie)}")
        bus.log(f"   Poids : {taille:.1f} Mo — encodée en "
                f"{int(elapsed // 60)}m {int(elapsed % 60):02d}s")
        bus.v_termine({
            "fichier": os.path.abspath(sortie),
            "dossier": dossier,
            "taille_mo": taille,
            "duree_calcul": elapsed,
            "duree_video": duree or 0.0,
        })

    except Exception as e:
        import traceback
        bus.log(traceback.format_exc())
        bus.v_erreur(str(e))
    finally:
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        if fichier_err is not None:
            try:
                fichier_err.close()
            except Exception:
                pass
            try:
                os.unlink(fichier_err.name)
            except Exception:
                pass


# =============================================================================
#  SECTION 4 ter — FUSION AUDIO ET CHAPITRAGE
# =============================================================================

def libelle_depuis_nom(chemin):
    """Propose un libellé de chapitre lisible à partir d'un nom de fichier :
    « 03_le-grand-depart.mp3 » devient « Le grand départ »."""
    nom = os.path.splitext(os.path.basename(chemin))[0]
    # Numérotation de tête : « 03 - », « 03. », « 03_ », « 03 »
    nom = re.sub(r'^\s*\d{1,3}\s*[-_.)\]]?\s*', '', nom)
    nom = nom.replace("_", " ").replace("-", " ")
    nom = re.sub(r'\s{2,}', ' ', nom).strip()
    if not nom:
        nom = os.path.splitext(os.path.basename(chemin))[0]
    return nom[:1].upper() + nom[1:]


def formater_horodate(secondes, forcer_heures=False):
    """Horodatage de chapitre. Les secondes sont TRONQUÉES (jamais arrondies au
    supérieur) pour qu'un chapitre ne démarre jamais après le début réel du
    segment."""
    total = max(0, int(secondes))
    h, reste = divmod(total, 3600)
    m, s = divmod(reste, 60)
    if h or forcer_heures:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def calculer_chapitres(segments, silence):
    """Calcule la position de chaque chapitre dans le fichier fusionné.

    Le silence inséré entre deux segments est compté AVANT le segment suivant :
    l'horodatage pointe donc sur le début réel de l'audio, pas sur le blanc qui
    le précède. Renvoie la liste des chapitres et la durée totale."""
    curseur = 0.0
    chapitres = []
    dernier = len(segments) - 1
    for i, seg in enumerate(segments):
        duree = float(seg.get("duree") or 0.0)
        chapitres.append({
            "debut": curseur,
            "fin": curseur + duree,
            "libelle": (seg.get("libelle") or "").strip() or f"Chapitre {i + 1}",
            "fichier": seg["chemin"],
            "duree": duree,
        })
        curseur += duree
        if i < dernier:
            curseur += max(0.0, float(silence))
    return chapitres, curseur


def formater_chapitres(chapitres, duree_totale, style):
    """Met en forme le chapitrage selon le style demandé."""
    forcer_h = duree_totale >= 3600
    lignes = []
    for ch in chapitres:
        t = formater_horodate(ch["debut"], forcer_h)
        libelle = ch["libelle"]
        if style == "youtube":
            lignes.append(f"{t} {libelle}")
        elif style == "complet":
            lignes.append(f"{formater_horodate(ch['debut'], True)} {libelle}")
        elif style == "libelle_puis_temps":
            lignes.append(f"{libelle} — {t}")
        elif style == "csv":
            lignes.append(f"{formater_horodate(ch['debut'], True)};{libelle}")
        else:
            lignes.append(f"{t} {libelle}")
    return "\n".join(lignes)


def verifier_regles_youtube(chapitres, duree_totale):
    """Contrôle les trois règles imposées par YouTube pour qu'un chapitrage
    soit reconnu. Renvoie une liste d'avertissements (vide si tout va bien)."""
    avertissements = []
    if not chapitres:
        return ["Aucun chapitre."]
    if int(chapitres[0]["debut"]) != 0:
        avertissements.append("Le premier chapitre doit commencer à 00:00.")
    if len(chapitres) < 3:
        avertissements.append(
            f"YouTube exige au moins 3 chapitres (il y en a {len(chapitres)}).")
    for i, ch in enumerate(chapitres):
        fin = chapitres[i + 1]["debut"] if i + 1 < len(chapitres) else duree_totale
        if fin - ch["debut"] < 10:
            avertissements.append(
                f"« {ch['libelle']} » dure moins de 10 s : YouTube ignorera "
                "tout le chapitrage.")
    return avertissements


def _echapper_ffmetadata(valeur):
    """Échappe les caractères réservés du format ffmetadata."""
    for c in ("\\", "=", ";", "#", "\n"):
        valeur = valeur.replace(c, "\\" + c if c != "\n" else "\\\n")
    return valeur


def ecrire_ffmetadata(chapitres, duree_totale, chemin):
    """Écrit un fichier de métadonnées ffmpeg décrivant les chapitres, en vue
    de les intégrer directement dans le fichier audio (MP3, M4A, MKV…)."""
    lignes = [";FFMETADATA1"]
    for i, ch in enumerate(chapitres):
        fin = chapitres[i + 1]["debut"] if i + 1 < len(chapitres) else duree_totale
        lignes += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={int(ch['debut'] * 1000)}",
            f"END={int(max(fin, ch['debut'] + 0.001) * 1000)}",
            f"title={_echapper_ffmetadata(ch['libelle'])}",
        ]
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    return chemin


# Conteneurs sachant stocker des chapitres en métadonnées.
CONTENEURS_CHAPITRES = {".mp3", ".m4a", ".m4b", ".mp4", ".mkv", ".ogg", ".opus", ".webm"}


def codec_pour_extension(ext, bitrate):
    """Options d'encodage déduites de l'extension du fichier de sortie."""
    ext = ext.lower()
    if ext == ".wav":
        return ["-c:a", "pcm_s16le"]
    if ext == ".flac":
        return ["-c:a", "flac"]
    if ext == ".mp3":
        return ["-c:a", "libmp3lame", "-b:a", bitrate]
    if ext in (".m4a", ".m4b", ".mp4", ".aac"):
        return ["-c:a", "aac", "-b:a", bitrate]
    if ext == ".opus":
        return ["-c:a", "libopus", "-b:a", bitrate]
    if ext == ".ogg":
        return ["-c:a", "libvorbis", "-b:a", bitrate]
    if ext == ".mkv":
        return ["-c:a", "libmp3lame", "-b:a", bitrate]
    return ["-c:a", "aac", "-b:a", bitrate]


def construire_commande_fusion(config, ffmpeg, segments):
    """Assemble la commande ffmpeg qui concatène les segments.

    On passe par le filtre `concat` (et non par le démuxeur concat) : il
    ré-échantillonne au vol, ce qui permet de mélanger des fichiers de formats,
    de fréquences et de nombres de canaux différents sans préparation."""
    sr = int(config["sample_rate"])
    layout = "mono" if config["mono"] else "stereo"
    silence = max(0.0, float(config["silence"]))
    lufs = float(config["lufs"])
    normalisation = config["normalisation"]
    dernier = len(segments) - 1

    cmd = [ffmpeg, "-y", "-hide_banner", "-nostats", "-loglevel", "error"]
    for seg in segments:
        cmd += ["-i", seg["chemin"]]

    format_commun = (f"aformat=sample_fmts=fltp:sample_rates={sr}:"
                     f"channel_layouts={layout}")
    filtres = []
    etiquettes = []
    for i in range(len(segments)):
        chaine = f"[{i}:a]"
        if normalisation == "segment":
            # loudnorm sort en 192 kHz : le aformat doit venir APRÈS lui,
            # sinon toute la chaîne repart sur une fréquence inattendue.
            chaine += f"loudnorm=I={lufs}:TP=-1.5:LRA=11,{format_commun}"
        else:
            chaine += format_commun
        if silence > 0 and i < dernier:
            chaine += f",apad=pad_dur={silence}"
        chaine += f"[a{i}]"
        filtres.append(chaine)
        etiquettes.append(f"[a{i}]")

    sortie_filtre = "out"
    concat = "".join(etiquettes) + f"concat=n={len(segments)}:v=0:a=1[out]"
    filtres.append(concat)
    if normalisation == "globale":
        filtres.append(f"[out]loudnorm=I={lufs}:TP=-1.5:LRA=11,{format_commun}[outn]")
        sortie_filtre = "outn"

    ext = os.path.splitext(config["sortie"])[1]
    cmd += [
        "-filter_complex", ";".join(filtres),
        "-map", f"[{sortie_filtre}]",
        "-ar", str(sr),
        "-ac", "1" if config["mono"] else "2",
    ]
    cmd += codec_pour_extension(ext, config["bitrate"])
    cmd += ["-progress", "pipe:1", config["sortie"]]
    return cmd


def integrer_chapitres(ffmpeg, fichier, chemin_meta, bus):
    """Réinjecte les chapitres dans le fichier fusionné par un simple remux
    (`-c copy`, donc sans réencodage ni perte). Opère sur un fichier temporaire
    et ne remplace l'original qu'en cas de succès : un échec ici ne peut jamais
    abîmer la fusion déjà réussie."""
    base, ext = os.path.splitext(fichier)
    if ext.lower() not in CONTENEURS_CHAPITRES:
        bus.log(f"   ⚠️  Le format {ext} ne gère pas les chapitres intégrés — "
                "étape ignorée (le fichier texte reste disponible).")
        return False
    temporaire = base + ".chapitres-tmp" + ext
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-i", fichier, "-i", chemin_meta,
           "-map_metadata", "1", "-map", "0:a", "-c", "copy", temporaire]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, **options_subprocess())
        os.replace(temporaire, fichier)
        bus.log("   ✓ Chapitres intégrés dans les métadonnées du fichier.")
        return True
    except subprocess.CalledProcessError as e:
        detail = e.stderr.decode("utf-8", errors="replace").strip()[-300:] if e.stderr else ""
        bus.log(f"   ⚠️  Intégration des chapitres impossible : {detail}")
    except Exception as e:
        bus.log(f"   ⚠️  Intégration des chapitres impossible : {e}")
    finally:
        if os.path.exists(temporaire):
            try:
                os.unlink(temporaire)
            except Exception:
                pass
    return False


def controler_fichier_ecrit(sortie, duree_attendue, ffprobe, bus):
    """Relit le fichier réellement présent sur le disque à la toute fin, une
    fois les chapitres intégrés, et compare sa durée à celle attendue.

    Un encodage peut « réussir » du point de vue de ffmpeg et pourtant laisser
    un fichier tronqué : disque plein, antivirus, ou dossier synchronisé dans
    le nuage (OneDrive, Google Drive) qui n'a pas fini de matérialiser le
    fichier. Sans ce contrôle, le problème ne se découvre qu'à la lecture."""
    duree_reelle = duree_media(sortie, ffprobe)
    if duree_reelle is None:
        bus.log("   ⚠️  Durée du fichier écrit illisible : vérifie-le à la lecture.")
        return None

    bus.log(f"   Durée relue sur le disque : "
            f"{formater_horodate(duree_reelle, True)} "
            f"(attendu {formater_horodate(duree_attendue, True)})")

    # 1 % de tolérance, au minimum 2 secondes : l'encodage MP3 ajoute quelques
    # dizaines de millisecondes de silence par segment.
    if abs(duree_reelle - duree_attendue) <= max(2.0, duree_attendue * 0.01):
        return duree_reelle

    bus.log("   ⚠️  ATTENTION : le fichier écrit ne fait pas la durée attendue.")
    if duree_reelle < duree_attendue:
        bus.log("      Le fichier est tronqué. Causes les plus fréquentes :")
        bus.log("      • dossier synchronisé dans le nuage (Google Drive, OneDrive,")
        bus.log("        Dropbox) : écris d'abord sur un disque local, puis copie ;")
        bus.log("      • espace disque insuffisant sur le volume de destination ;")
        bus.log("      • antivirus qui interrompt l'écriture d'un gros fichier.")
    return duree_reelle


def worker_fusion(config, bus, stop_event):
    """Thread de travail de la fusion : mesure des durées, concaténation par
    ffmpeg, chapitrage, écritures annexes et contrôle final du fichier écrit."""
    t0 = time.time()
    proc = None
    fichier_err = None
    import tempfile

    try:
        segments = config["segments"]
        sortie = config["sortie"]

        if len(segments) < 2:
            bus.f_erreur("Ajoute au moins deux fichiers audio à fusionner.")
            return
        if not sortie:
            bus.f_erreur("Indique l'emplacement et le nom du fichier fusionné.")
            return
        for seg in segments:
            if not os.path.exists(seg["chemin"]):
                bus.f_erreur(f"Fichier introuvable : {seg['chemin']}")
                return

        ffmpeg = trouver_ffmpeg(config.get("ffmpeg_path"))
        if ffmpeg is None:
            bus.f_erreur(
                "ffmpeg est introuvable : la fusion en dépend entièrement.\n\n"
                "Installe-le (winget install --id=Gyan.FFmpeg -e) ou indique "
                "son chemin dans l'onglet « Export MP3 ».")
            return
        ffprobe = trouver_ffprobe(ffmpeg)

        os.makedirs(os.path.dirname(os.path.abspath(sortie)), exist_ok=True)

        # --- Mesure des durées (source du chapitrage) ---
        bus.f_status("Analyse des fichiers…")
        for seg in segments:
            if stop_event.is_set():
                bus.f_erreur("Fusion annulée.")
                return
            duree = duree_media(seg["chemin"], ffprobe)
            if duree is None:
                bus.f_erreur(
                    f"Durée illisible pour {os.path.basename(seg['chemin'])}.\n"
                    "Le chapitrage serait faux : fusion interrompue.")
                return
            seg["duree"] = duree

        chapitres, duree_totale = calculer_chapitres(segments, config["silence"])
        texte_chapitres = formater_chapitres(chapitres, duree_totale,
                                             config["style_chapitres"])

        bus.log("═" * 50)
        bus.log(f"🔗 Fusion de {len(segments)} fichiers")
        bus.log(f"   Sortie  : {os.path.abspath(sortie)}")
        bus.log(f"   Durée   : {int(duree_totale // 60)} min "
                f"{int(duree_totale % 60)} sec")
        bus.log(f"   Silence : {config['silence']} s entre segments  •  "
                f"{config['sample_rate']} Hz  •  "
                f"{'mono' if config['mono'] else 'stéréo'}")
        etiquette_norm = {"aucune": "aucune",
                          "segment": f"par segment ({config['lufs']} LUFS)",
                          "globale": f"globale ({config['lufs']} LUFS)"}
        bus.log(f"   Normalisation : {etiquette_norm.get(config['normalisation'], '—')}")

        for avert in verifier_regles_youtube(chapitres, duree_totale):
            bus.log(f"   ⚠️  YouTube : {avert}")

        # Le chapitrage est transmis à l'interface AVANT l'encodage : même si
        # la fusion échoue ensuite, le texte reste exploitable.
        bus.f_chapitres(texte_chapitres)

        cmd = construire_commande_fusion(config, ffmpeg, segments)
        bus.log("   Commande : " + " ".join(
            f'"{a}"' if " " in a else a for a in cmd))
        bus.f_status("Fusion en cours…")

        fichier_err = tempfile.NamedTemporaryFile(
            prefix="qwen_fusion_", suffix=".log", delete=False)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=fichier_err, bufsize=1,
            text=True, encoding="utf-8", errors="replace", **options_subprocess())

        annule = False
        for ligne in proc.stdout:
            if stop_event.is_set() and not annule:
                annule = True
                bus.log("⏹ Arrêt demandé — interruption de ffmpeg.")
                try:
                    proc.terminate()
                except Exception:
                    pass
                break
            ligne = ligne.strip()
            if ligne.startswith("out_time="):
                secondes = _secondes_depuis_horodate(ligne.split("=", 1)[1].strip())
                if secondes is None:
                    continue
                if duree_totale > 0:
                    bus.f_progress(min(100.0, secondes / duree_totale * 100.0))
                    bus.f_status(f"Fusion… {int(secondes // 60)}m "
                                 f"{int(secondes % 60):02d}s / "
                                 f"{int(duree_totale // 60)}m "
                                 f"{int(duree_totale % 60):02d}s")

        proc.wait()
        try:
            fichier_err.close()
        except Exception:
            pass

        if annule or stop_event.is_set():
            bus.f_erreur("Fusion annulée. Le fichier de sortie est incomplet.")
            return

        if proc.returncode != 0:
            detail = ""
            try:
                with open(fichier_err.name, encoding="utf-8", errors="replace") as fp:
                    detail = fp.read().strip()[-800:]
            except Exception:
                pass
            bus.log(detail or "(aucun détail fourni par ffmpeg)")
            bus.f_erreur(f"ffmpeg a échoué (code {proc.returncode}).\n\n"
                         f"{detail[-400:] if detail else 'Voir le journal.'}")
            return

        bus.f_progress(100.0)
        taille = os.path.getsize(sortie) / (1024 * 1024) if os.path.exists(sortie) else 0.0
        bus.log(f"   ✓ Fichier fusionné : {os.path.abspath(sortie)}  "
                f"({taille:.1f} Mo)")
        # --- Écritures annexes ---
        chemin_txt = None
        if config["ecrire_txt"]:
            chemin_txt = os.path.splitext(sortie)[0] + "-chapitres.txt"
            try:
                with open(chemin_txt, "w", encoding="utf-8") as f:
                    f.write(texte_chapitres + "\n")
                bus.log(f"   ✓ Chapitrage : {os.path.abspath(chemin_txt)}")
            except Exception as e:
                bus.log(f"   ⚠️  Écriture du chapitrage impossible : {e}")
                chemin_txt = None

        if config["integrer_chapitres"]:
            bus.f_status("Intégration des chapitres…")
            meta = os.path.splitext(sortie)[0] + ".ffmetadata.txt"
            try:
                ecrire_ffmetadata(chapitres, duree_totale, meta)
                integrer_chapitres(ffmpeg, sortie, meta, bus)
            finally:
                if os.path.exists(meta):
                    try:
                        os.unlink(meta)
                    except Exception:
                        pass

        elapsed = time.time() - t0
        duree_ecrite = controler_fichier_ecrit(sortie, duree_totale, ffprobe, bus)
        bus.f_termine({
            "fichier": os.path.abspath(sortie),
            "dossier": os.path.dirname(os.path.abspath(sortie)),
            "taille_mo": taille,
            "duree_totale": duree_totale,
            "duree_calcul": elapsed,
            "nb_segments": len(segments),
            "chapitres": texte_chapitres,
            "fichier_txt": chemin_txt,
            "duree_ecrite": duree_ecrite,
        })

    except Exception as e:
        import traceback
        bus.log(traceback.format_exc())
        bus.f_erreur(str(e))
    finally:
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        if fichier_err is not None:
            try:
                fichier_err.close()
            except Exception:
                pass
            try:
                os.unlink(fichier_err.name)
            except Exception:
                pass


# =============================================================================
#  SECTION 5 — INTERFACE GRAPHIQUE
# =============================================================================

PRESETS_EMOTION = [
    "Neutral, calm, measured, natural pace.",
    "Neutral, clear, engaged, conversational.",
    "Neutral, documentary narration, even tone.",
    "Warm, storytelling, slightly slower pace.",
    "Energetic, enthusiastic, dynamic delivery.",
    "Serious, authoritative, deliberate.",
]

LANGUES = [
    "French", "English", "Chinese", "Spanish", "German",
    "Italian", "Japanese", "Korean", "Portuguese", "Russian",
]

BITRATES = ["96k", "128k", "160k", "192k", "256k", "320k"]

# --- Montage vidéo -----------------------------------------------------------
# Libellé affiché → valeur passée au moteur ("source" = taille de l'image).
RESOLUTIONS = {
    "Taille de l'image d'origine": "source",
    "1920 × 1080 — 16:9 (YouTube)": "1920x1080",
    "1280 × 720 — 16:9 (léger)": "1280x720",
    "2560 × 1440 — 16:9 (2K)": "2560x1440",
    "3840 × 2160 — 16:9 (4K)": "3840x2160",
    "1080 × 1080 — 1:1 (carré)": "1080x1080",
    "1080 × 1920 — 9:16 (Shorts / Reels)": "1080x1920",
}

FPS_VIDEO = ["1", "2", "5", "10", "15", "24", "25", "30"]

PRESETS_X264 = ["ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow"]

FORMATS_VIDEO = [("MP4 (H.264 / AAC)", "*.mp4"),
                 ("Matroska", "*.mkv"),
                 ("QuickTime", "*.mov"),
                 ("Tous les fichiers", "*.*")]

EXTENSIONS_IMAGE = "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"
EXTENSIONS_AUDIO = "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.opus"

# --- Fusion audio -----------------------------------------------------------
FORMATS_FUSION = [("MP3", "*.mp3"), ("WAV", "*.wav"), ("M4A / AAC", "*.m4a"),
                  ("FLAC", "*.flac"), ("Opus", "*.opus"),
                  ("Tous les fichiers", "*.*")]

NORMALISATIONS = {
    "Aucune (conserve les niveaux d'origine)": "aucune",
    "Par segment — harmonise les fichiers entre eux": "segment",
    "Globale — sur le fichier final uniquement": "globale",
}

# Cibles de loudness usuelles selon la plateforme de diffusion.
CIBLES_LUFS = ["-14 (YouTube, Spotify)", "-16 (podcast, Apple)", "-18", "-23 (EBU R128 TV)"]

STYLES_CHAPITRES = {
    "YouTube — 00:00 Libellé": "youtube",
    "Horodatage complet — 00:00:00 Libellé": "complet",
    "Libellé puis horodatage": "libelle_puis_temps",
    "CSV — 00:00:00;Libellé": "csv",
}

FREQUENCES_FUSION = ["22050", "24000", "44100", "48000"]

# Dossiers de rangement dont le nom ne dit rien de l'œuvre : quand les segments
# y résident, le nom du fichier fusionné est cherché un cran au-dessus.
DOSSIERS_GENERIQUES = {
    "parties", "partie", "parts", "part", "chapitres", "chapters", "segments",
    "audio", "audios", "son", "sons", "mp3", "wav", "flac", "m4a",
    "sortie", "sorties", "output", "outputs", "out", "rendu", "rendus",
    "export", "exports", "temp", "tmp", "downloads", "téléchargements",
}

# Transcription de référence proposée par défaut (modifiable dans l'interface).
TEXTE_REF_DEFAUT = ""          # à remplir avec la transcription de VOTRE extrait

CONFIG_DEFAUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "qwen_tts_studio_config.json")


class Tooltip:
    """Info-bulle légère pour les champs de l'interface."""
    def __init__(self, widget, texte):
        self.widget = widget
        self.texte = texte
        self.tip = None
        widget.bind("<Enter>", self._afficher)
        widget.bind("<Leave>", self._cacher)

    def _afficher(self, _=None):
        if self.tip or not self.texte:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tip, text=self.texte, justify="left",
                       background="#2b2b3a", foreground="#e8e8f0",
                       relief="solid", borderwidth=1, padx=8, pady=5,
                       font=("Segoe UI", 9), wraplength=320)
        lbl.pack()

    def _cacher(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class TTSStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Qwen3-TTS Voice Studio")
        self.geometry("1240x820")
        self.minsize(1040, 700)

        # État interne
        self.file_msgs = queue.Queue()
        self.bus = Bus(self.file_msgs)
        self.stop_event = threading.Event()
        self.worker = None
        self.model_cache = {}            # persiste le modèle entre les rendus
        self.dernier_dossier = None
        self.dernier_fichier = None
        self.dernier_projet = None       # dossier « _projet » du dernier rendu
        # Montage vidéo : thread et signal d'arrêt distincts de ceux du TTS
        self.stop_video = threading.Event()
        self.worker_video = None
        self.derniere_video = None
        # Fusion audio : segments à concaténer et thread dédié
        self.stop_fusion = threading.Event()
        self.worker_fusion = None
        self.segments = []               # [{chemin, libelle, duree}]
        self._editeur_libelle = None     # champ d'édition en ligne actif
        self.dernier_fusionne = None
        # Le chemin de sortie proposé automatiquement reste ajustable tant que
        # l'utilisateur n'a pas saisi le sien.
        self._sortie_fusion_auto = True
        self._sortie_auto_en_cours = False
        self.taches = set()              # {"tts", "video", "fusion"} en cours
        self.t_debut = None
        self.pret = False           # passe à True quand les biblio sont chargées

        self._init_styles()
        self._init_variables()
        self._construire_ui()

        # Chargement automatique de la dernière configuration
        if os.path.exists(CONFIG_DEFAUT):
            try:
                self._charger_config(CONFIG_DEFAUT, silencieux=True)
            except Exception:
                pass

        self.protocol("WM_DELETE_WINDOW", self._fermer)
        self.after(80, self._pomper_messages)
        # Préchargement des bibliothèques natives SUR LE THREAD PRINCIPAL,
        # juste après le premier affichage de la fenêtre (voir note Windows).
        self.after(300, self._precharger_bibliotheques)

    # ---------------------------------------------------------------- styles
    def _init_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        BG = "#f4f4f8"
        self.configure(bg=BG)
        style.configure(".", background=BG, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG)
        style.configure("TLabelframe", background=BG)
        style.configure("TLabelframe.Label", background=BG,
                        font=("Segoe UI Semibold", 10), foreground="#3a3a5a")
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", padding=(14, 7), font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 16),
                        foreground="#2a2a4a")
        style.configure("Sub.TLabel", foreground="#6a6a85", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 11),
                        foreground="white", background="#4a55c8", padding=8)
        style.map("Accent.TButton",
                  background=[("active", "#3a44b0"), ("disabled", "#a8accf")])
        style.configure("Stop.TButton", font=("Segoe UI Semibold", 11),
                        foreground="white", background="#c8504a", padding=8)
        style.map("Stop.TButton",
                  background=[("active", "#b03a3a"), ("disabled", "#d8a8a8")])
        style.configure("TButton", padding=5)

    # ------------------------------------------------------------- variables
    def _init_variables(self):
        self.v_ref_audio = tk.StringVar()
        self.v_text_file = tk.StringVar()
        self.v_output_dir = tk.StringVar()
        self.v_output_base = tk.StringVar(value="rendu")
        self.v_horodater = tk.BooleanVar(value=False)
        self.v_instruction = tk.StringVar(value=PRESETS_EMOTION[0])
        self.v_language = tk.StringVar(value="French")
        self.v_max_chars = tk.IntVar(value=350)
        self.v_taille_lot = tk.IntVar(value=TAILLE_LOT)
        # --- Livre audio, qualité, dialogues ---
        self.v_qc = tk.BooleanVar(value=False)
        self.v_srt = tk.BooleanVar(value=True)
        self.v_dossier_voix = tk.StringVar()
        self.v_livre = tk.BooleanVar(value=False)
        self.v_livre_titre = tk.StringVar()
        self.v_livre_auteur = tk.StringVar()
        self.v_livre_couverture = tk.StringVar()
        self.v_garder_morceaux = tk.BooleanVar(value=True)
        self.v_sil_virgule = tk.DoubleVar(value=0.15)
        self.v_sil_point = tk.DoubleVar(value=0.40)
        self.v_sil_para = tk.DoubleVar(value=0.70)
        self.v_target_peak = tk.DoubleVar(value=-3.0)
        self.v_freq_supp = tk.StringVar(value="44100")
        self.v_export_mp3 = tk.BooleanVar(value=True)
        self.v_mp3_bitrate = tk.StringVar(value="192k")
        self.v_mp3_mono = tk.BooleanVar(value=True)
        self.v_mp3_source = tk.StringVar(value="auto")
        self.v_ffmpeg_path = tk.StringVar()
        self.v_model_id = tk.StringVar(value="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        self.v_device = tk.StringVar(value="auto")
        self.v_hf_token = tk.StringVar(value=os.environ.get("HF_TOKEN", ""))
        self.v_remember_token = tk.BooleanVar(value=False)
        self.v_progress = tk.StringVar(value="Prêt.")

        # --- Montage vidéo (image fixe + audio) ---
        self.v_video_image = tk.StringVar()
        self.v_video_audio = tk.StringVar()
        self.v_video_sortie = tk.StringVar()
        self.v_video_resolution = tk.StringVar(value="1920 × 1080 — 16:9 (YouTube)")
        self.v_video_fps = tk.StringVar(value="25")
        self.v_video_crf = tk.IntVar(value=20)
        self.v_video_preset = tk.StringVar(value="medium")
        self.v_video_fond = tk.StringVar(value="black")
        self.v_video_audio_copie = tk.BooleanVar(value=False)
        self.v_video_audio_bitrate = tk.StringVar(value="192k")
        self.v_video_auto = tk.BooleanVar(value=False)

        # --- Fusion audio et chapitrage ---
        self.v_fusion_sortie = tk.StringVar()
        self.v_fusion_silence = tk.DoubleVar(value=1.0)
        self.v_fusion_sample_rate = tk.StringVar(value="44100")
        self.v_fusion_mono = tk.BooleanVar(value=True)
        self.v_fusion_bitrate = tk.StringVar(value="192k")
        self.v_fusion_normalisation = tk.StringVar(
            value="Par segment — harmonise les fichiers entre eux")
        self.v_fusion_lufs = tk.StringVar(value="-16 (podcast, Apple)")
        self.v_fusion_style = tk.StringVar(value="YouTube — 00:00 Libellé")
        self.v_fusion_txt = tk.BooleanVar(value=True)
        self.v_fusion_integrer = tk.BooleanVar(value=True)
        self.v_fusion_info = tk.StringVar(value="Aucun fichier.")

    # -------------------------------------------------------------------- UI
    def _construire_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # En-tête
        header = ttk.Frame(self, padding=(16, 12, 16, 6))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="🎙  Qwen3-TTS Voice Studio",
                  style="Header.TLabel").pack(side="left")
        self.lbl_modele = ttk.Label(header, text="● Modèle non chargé",
                                    style="Sub.TLabel", foreground="#a05a5a")
        self.lbl_modele.pack(side="right")

        # Panneau gauche : onglets de réglages
        gauche = ttk.Frame(self, padding=(14, 6, 8, 6))
        gauche.grid(row=1, column=0, sticky="nsew")
        nb = ttk.Notebook(gauche, width=620)
        nb.pack(fill="both", expand=True)
        self._onglet_entrees(nb)
        self._onglet_voix(nb)
        self._onglet_livre(nb)
        self._onglet_export(nb)
        self._onglet_video(nb)
        self._onglet_fusion(nb)
        self._onglet_systeme(nb)

        # Panneau droit : console de log
        droite = ttk.Frame(self, padding=(8, 6, 14, 6))
        droite.grid(row=1, column=1, sticky="nsew")
        droite.rowconfigure(1, weight=1)
        droite.columnconfigure(0, weight=1)

        barre_log = ttk.Frame(droite)
        barre_log.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(barre_log, text="Journal", style="TLabelframe.Label").pack(side="left")
        ttk.Button(barre_log, text="Effacer", command=self._effacer_log).pack(side="right")
        ttk.Button(barre_log, text="Aperçu du découpage",
                   command=self._apercu_decoupage).pack(side="right", padx=4)

        self.log = scrolledtext.ScrolledText(
            droite, wrap="word", font=("Consolas", 9),
            background="#1e1e2a", foreground="#d8d8e8",
            insertbackground="#d8d8e8", relief="flat", padx=10, pady=8)
        self.log.grid(row=1, column=0, sticky="nsew")
        self.log.configure(state="disabled")

        # Barre d'action en bas
        action = ttk.Frame(self, padding=(16, 8, 16, 12))
        action.grid(row=2, column=0, columnspan=2, sticky="ew")
        action.columnconfigure(2, weight=1)

        self.btn_generer = ttk.Button(action, text="▶  Générer l'audio",
                                      style="Accent.TButton", command=self._lancer,
                                      state="disabled")
        self.btn_generer.grid(row=0, column=0, padx=(0, 6))
        self.btn_stop = ttk.Button(action, text="⏹  Arrêter",
                                   style="Stop.TButton", command=self._arreter,
                                   state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=(0, 12))

        self.progress = ttk.Progressbar(action, mode="determinate")
        self.progress.grid(row=0, column=2, sticky="ew", padx=8)
        ttk.Label(action, textvariable=self.v_progress,
                  style="Sub.TLabel").grid(row=0, column=3, padx=8)

        self.btn_ouvrir = ttk.Button(action, text="📂 Dossier",
                                     command=self._ouvrir_dossier, state="disabled")
        self.btn_ouvrir.grid(row=0, column=4, padx=4)
        self.btn_lire = ttk.Button(action, text="🔊 Lire",
                                   command=self._lire_rendu, state="disabled")
        self.btn_lire.grid(row=0, column=5, padx=4)

        self._log_init("Bienvenue dans Qwen3-TTS Voice Studio.")
        self._log_init("Sélectionne une voix de référence et un texte, puis lance la génération.")

    def _onglet_entrees(self, nb):
        f = ttk.Frame(nb, padding=14)
        nb.add(f, text="  Entrées  ")
        f.columnconfigure(1, weight=1)
        r = 0

        ttk.Label(f, text="Voix de référence (à cloner)").grid(row=r, column=0, sticky="w", pady=(0, 2), columnspan=2)
        r += 1
        e = ttk.Entry(f, textvariable=self.v_ref_audio)
        e.grid(row=r, column=0, columnspan=2, sticky="ew")
        Tooltip(e, "Fichier audio de la voix à imiter (WAV recommandé, ~10–30 s, propre).")
        b = ttk.Button(f, text="Parcourir…", command=self._choisir_ref)
        b.grid(row=r, column=2, padx=(6, 0))
        r += 1

        ligne = ttk.Frame(f)
        ligne.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(12, 2))
        ttk.Label(ligne, text="Transcription EXACTE de la voix de référence").pack(side="left")
        self.btn_transcrire = ttk.Button(ligne, text="🎧 Transcrire automatiquement",
                                         command=self._transcrire_ref)
        self.btn_transcrire.pack(side="right")
        Tooltip(self.btn_transcrire,
                "Whisper écoute la voix de référence et remplit ce champ.\n"
                "Relisez le résultat : une transcription exacte améliore le clonage.\n"
                "Premier usage : téléchargement du modèle Whisper (~1,6 Go).")
        r += 1
        self.txt_ref = tk.Text(f, height=5, wrap="word", font=("Segoe UI", 10),
                               relief="solid", borderwidth=1, padx=6, pady=4)
        self.txt_ref.grid(row=r, column=0, columnspan=3, sticky="ew")
        self.txt_ref.insert("1.0", TEXTE_REF_DEFAUT)
        Tooltip(self.txt_ref, "Texte prononcé MOT POUR MOT dans l'extrait de référence.\n"
                              "Plus c'est fidèle, meilleur est le conditionnement de la voix.")
        r += 1

        ttk.Label(f, text="Fichier texte à lire (.txt, .md, .docx)").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(12, 2))
        r += 1
        e2 = ttk.Entry(f, textvariable=self.v_text_file)
        e2.grid(row=r, column=0, columnspan=2, sticky="ew")
        Tooltip(e2, "Texte source à vocaliser. Les fichiers Word (.docx) sont pris en charge.")
        ttk.Button(f, text="Parcourir…", command=self._choisir_texte).grid(row=r, column=2, padx=(6, 0))
        r += 1

        sep = ttk.Separator(f)
        sep.grid(row=r, column=0, columnspan=3, sticky="ew", pady=14)
        r += 1

        ttk.Label(f, text="Dossier de sortie").grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 2))
        r += 1
        ttk.Entry(f, textvariable=self.v_output_dir).grid(row=r, column=0, columnspan=2, sticky="ew")
        ttk.Button(f, text="Parcourir…", command=self._choisir_dossier).grid(row=r, column=2, padx=(6, 0))
        r += 1

        ttk.Label(f, text="Nom de base du rendu").grid(row=r, column=0, sticky="w", pady=(10, 2))
        ttk.Checkbutton(f, text="Horodater (ne jamais écraser)",
                        variable=self.v_horodater).grid(row=r, column=1, columnspan=2, sticky="e", pady=(10, 2))
        r += 1
        ttk.Entry(f, textvariable=self.v_output_base).grid(row=r, column=0, columnspan=3, sticky="ew")

    def _onglet_voix(self, nb):
        f = ttk.Frame(nb, padding=14)
        nb.add(f, text="  Voix & Découpage  ")
        f.columnconfigure(1, weight=1)
        r = 0

        ttk.Label(f, text="Ton / émotion (instruction)").grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 2))
        r += 1
        cb = ttk.Combobox(f, textvariable=self.v_instruction, values=PRESETS_EMOTION)
        cb.grid(row=r, column=0, columnspan=2, sticky="ew")
        Tooltip(cb, "Sans effet avec les modèles Base (clonage vocal) : le ton vient "
                    "de l'enregistrement de référence. Utile seulement avec un modèle "
                    "CustomVoice / VoiceDesign.")
        r += 1

        ttk.Label(f, text="Langue de synthèse").grid(row=r, column=0, sticky="w", pady=(12, 2))
        r += 1
        ttk.Combobox(f, textvariable=self.v_language, values=LANGUES,
                     state="readonly").grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1

        ttk.Separator(f).grid(row=r, column=0, columnspan=2, sticky="ew", pady=14)
        r += 1

        def champ_num(label, var, hint):
            nonlocal r
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=4)
            sp = ttk.Entry(f, textvariable=var, width=12)
            sp.grid(row=r, column=1, sticky="e", pady=4)
            Tooltip(sp, hint)
            r += 1

        champ_num("Taille max d'un chunk (caractères)", self.v_max_chars,
                  "350 = bon compromis. En cas de saturation VRAM, descends à 300 ou 250.")
        champ_num("Phrases générées en parallèle (lot)", self.v_taille_lot,
                  "1 = ancien comportement. 4 à 8 accélère nettement sur GPU. "
                  "Si la VRAM sature, le lot est repris phrase par phrase.")
        champ_num("Silence après virgule (s)", self.v_sil_virgule, "Pause courte après , - – — :")
        champ_num("Silence après point (s)", self.v_sil_point, "Pause après . ! ? ;")
        champ_num("Silence entre paragraphes (s)", self.v_sil_para, "Pause longue entre deux paragraphes.")
        champ_num("Niveau cible (dBFS)", self.v_target_peak,
                  "-3 = fort sans saturation. -6 à -9 pour davantage de marge.")

        ttk.Label(f, text="Fréquence supplémentaire (Hz)").grid(row=r, column=0, sticky="w", pady=(8, 4))
        ttk.Combobox(f, textvariable=self.v_freq_supp, values=["Aucune", "44100", "48000"],
                     state="readonly", width=10).grid(row=r, column=1, sticky="e", pady=(8, 4))

    def _onglet_livre(self, nb):
        f = ttk.Frame(nb, padding=14)
        nb.add(f, text="  Livre & Qualité  ")
        f.columnconfigure(1, weight=1)
        r = 0

        # --- Qualité ---
        ttk.Label(f, text="Qualité", style="TLabelframe.Label").grid(
            row=r, column=0, columnspan=3, sticky="w")
        r += 1
        cb = ttk.Checkbutton(f, text="Contrôle qualité automatique (Whisper)",
                             variable=self.v_qc)
        cb.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 0))
        Tooltip(cb, "Après la génération, Whisper réécoute chaque phrase et la compare\n"
                    "au texte. Les phrases mal lues (mots sautés, répétitions, boucles)\n"
                    "sont régénérées automatiquement, deux fois au plus.\n"
                    "Coût : environ 10 à 20 % de temps en plus, ~1,6 Go de VRAM.")
        r += 1
        cb = ttk.Checkbutton(f, text="Créer les sous-titres (.srt)", variable=self.v_srt)
        cb.grid(row=r, column=0, columnspan=3, sticky="w", pady=(2, 0))
        Tooltip(cb, "Fichier de sous-titres synchronisés, à joindre à une vidéo YouTube.")
        r += 1

        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=12)
        r += 1

        # --- Dialogues ---
        ttk.Label(f, text="Dialogues à plusieurs voix", style="TLabelframe.Label").grid(
            row=r, column=0, columnspan=3, sticky="w")
        r += 1
        ttk.Label(f, text="Dossier des voix").grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 2))
        r += 1
        e = ttk.Entry(f, textvariable=self.v_dossier_voix)
        e.grid(row=r, column=0, columnspan=2, sticky="ew")
        Tooltip(e, "Dans le texte, commencez un paragraphe par [Marie] pour le faire lire\n"
                   "avec Marie.wav de ce dossier (transcription dans Marie.txt, créée\n"
                   "automatiquement par Whisper si elle manque). Sans balise ou avec\n"
                   "[Narrateur] : voix principale de l'onglet Entrées.")
        ttk.Button(f, text="Parcourir…", command=self._choisir_dossier_voix).grid(
            row=r, column=2, padx=(6, 0))
        r += 1
        ttk.Label(f, text="Exemple :  [Marie] Bonjour, dit-elle.   →   lu avec Marie.wav",
                  style="Sub.TLabel").grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 0))
        r += 1

        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=12)
        r += 1

        # --- Livre audio ---
        ttk.Label(f, text="Livre audio", style="TLabelframe.Label").grid(
            row=r, column=0, columnspan=3, sticky="w")
        r += 1
        cb = ttk.Checkbutton(f, text="Créer un livre audio M4B (un chapitre par titre du document)",
                             variable=self.v_livre)
        cb.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 2))
        Tooltip(cb, "Les chapitres suivent les titres du document : styles « Titre 1 »\n"
                    "de Word, ou lignes « # » en Markdown. Produit aussi un fichier\n"
                    "-chapitres.txt prêt à coller dans une description YouTube.")
        r += 1
        ttk.Label(f, text="Titre").grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(f, textvariable=self.v_livre_titre).grid(row=r, column=1, columnspan=2, sticky="ew", pady=3)
        r += 1
        ttk.Label(f, text="Auteur").grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(f, textvariable=self.v_livre_auteur).grid(row=r, column=1, columnspan=2, sticky="ew", pady=3)
        r += 1
        ttk.Label(f, text="Couverture").grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(f, textvariable=self.v_livre_couverture).grid(row=r, column=1, sticky="ew", pady=3)
        ttk.Button(f, text="Parcourir…", command=self._choisir_couverture).grid(
            row=r, column=2, padx=(6, 0), pady=3)
        r += 1

        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=12)
        r += 1

        # --- Correction ---
        ttk.Label(f, text="Correction", style="TLabelframe.Label").grid(
            row=r, column=0, columnspan=3, sticky="w")
        r += 1
        cb = ttk.Checkbutton(f, text="Conserver les morceaux pour pouvoir corriger des phrases",
                             variable=self.v_garder_morceaux)
        cb.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 2))
        Tooltip(cb, "Chaque phrase est gardée dans un dossier « …_projet » à côté du rendu.\n"
                    "Vous pourrez ensuite régénérer une phrase sans tout relancer.")
        r += 1
        self.btn_corriger = ttk.Button(f, text="✎ Corriger des phrases…",
                                       command=self._ouvrir_correction)
        self.btn_corriger.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _choisir_dossier_voix(self):
        chemin = filedialog.askdirectory(title="Dossier contenant les voix (Nom.wav)")
        if chemin:
            self.v_dossier_voix.set(chemin)

    def _choisir_couverture(self):
        chemin = filedialog.askopenfilename(
            title="Choisir la couverture du livre audio",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("Tous les fichiers", "*.*")])
        if chemin:
            self.v_livre_couverture.set(chemin)

    # ------------------------------------------------------- transcription
    def _transcrire_ref(self):
        if self.taches:
            return
        if not self.pret:
            messagebox.showinfo("Patiente", "Les bibliothèques sont encore en cours de chargement.")
            return
        chemin = self.v_ref_audio.get().strip()
        if not chemin or not os.path.exists(chemin):
            messagebox.showwarning("Voix manquante", "Sélectionne d'abord une voix de référence.")
            return
        self.taches.add("transcription")
        self._maj_boutons()
        self._log("🎧 Transcription de la voix de référence par Whisper…")
        threading.Thread(target=worker_transcription,
                         args=(chemin, self.v_language.get(), self.v_device.get(),
                               self.bus, self.model_cache),
                         daemon=True).start()

    # ---------------------------------------------------------- correction
    def _ouvrir_correction(self):
        if self.taches:
            return
        dossier = self.dernier_projet
        if not dossier or not os.path.isfile(os.path.join(dossier, "projet.json")):
            chemin = filedialog.askopenfilename(
                title="Ouvrir le projet d'un rendu (dossier …_projet)",
                filetypes=[("Projet de rendu", "projet.json")])
            if not chemin:
                return
            dossier = os.path.dirname(chemin)
        try:
            with open(os.path.join(dossier, "projet.json"), encoding="utf-8") as fp:
                projet = json.load(fp)
        except Exception as e:
            messagebox.showerror("Correction", f"Projet illisible :\n{e}")
            return

        fen = tk.Toplevel(self)
        fen.title("Corriger des phrases — " + os.path.basename(projet["sortie"]["wav"]))
        fen.geometry("980x560")
        fen.transient(self)
        ttk.Label(fen, padding=(12, 10, 12, 4),
                  text="Sélectionnez les phrases à refaire (Ctrl ou Maj pour en choisir plusieurs). "
                       "En rouge : phrases jugées douteuses par le contrôle qualité.").pack(anchor="w")
        cadre = ttk.Frame(fen, padding=(12, 0, 12, 0))
        cadre.pack(fill="both", expand=True)
        arbre = ttk.Treeview(cadre, columns=("n", "voix", "score", "texte"),
                             show="headings", selectmode="extended")
        for col, titre, larg in (("n", "N°", 50), ("voix", "Voix", 90),
                                 ("score", "Contrôle", 70), ("texte", "Texte", 720)):
            arbre.heading(col, text=titre)
            arbre.column(col, width=larg, stretch=(col == "texte"),
                         anchor="w" if col == "texte" else "center")
        defil = ttk.Scrollbar(cadre, orient="vertical", command=arbre.yview)
        arbre.configure(yscrollcommand=defil.set)
        arbre.pack(side="left", fill="both", expand=True)
        defil.pack(side="right", fill="y")
        arbre.tag_configure("douteux", foreground="#b03030")
        for k, c in enumerate(projet["chunks"]):
            score = c.get("score")
            arbre.insert("", "end", iid=str(k), values=(
                k + 1, c.get("voix") or "—",
                "" if score is None else f"{score:.0%}", c["texte"]),
                tags=("douteux",) if (score is not None and score < SEUIL_QC) else ())

        def selection():
            return sorted(int(i) for i in arbre.selection())

        def ecouter():
            sel = selection()
            if sel:
                c = projet["chunks"][sel[0]]
                chemin = os.path.join(dossier, "morceaux", c.get("fichier") or "")
                if c.get("fichier") and os.path.exists(chemin):
                    self._ouvrir_chemin(chemin)

        def douteuses():
            arbre.selection_set([str(k) for k, c in enumerate(projet["chunks"])
                                 if c.get("score") is not None and c["score"] < SEUIL_QC])

        def regenerer():
            sel = selection()
            if not sel:
                messagebox.showinfo("Correction", "Sélectionnez au moins une phrase.", parent=fen)
                return
            fen.destroy()
            self._lancer_correction(dossier, sel)

        bas = ttk.Frame(fen, padding=12)
        bas.pack(fill="x")
        ttk.Button(bas, text="▶ Écouter", command=ecouter).pack(side="left")
        ttk.Button(bas, text="Sélectionner les douteuses", command=douteuses).pack(side="left", padx=6)
        ttk.Button(bas, text="Fermer", command=fen.destroy).pack(side="right")
        ttk.Button(bas, text="🔁 Régénérer la sélection", style="Accent.TButton",
                   command=regenerer).pack(side="right", padx=6)
        arbre.bind("<Double-1>", lambda _e: ecouter())

    def _lancer_correction(self, dossier, indices):
        if self.taches or not self.pret:
            return
        config = self._construire_config()
        self.stop_event.clear()
        self.taches.add("tts")
        self._maj_boutons()
        self.progress.config(value=0, maximum=max(1, len(indices)))
        self.t_debut = time.time()
        self._tic_chrono()
        self._log("─" * 50)
        self._log(f"✎ Correction de {len(indices)} phrase(s) : n° "
                  + ", ".join(str(k + 1) for k in indices[:30]))
        self.worker = threading.Thread(
            target=worker_correction,
            args=(dossier, indices, config, self.bus, self.stop_event, self.model_cache),
            daemon=True)
        self.worker.start()

    def _onglet_export(self, nb):
        f = ttk.Frame(nb, padding=14)
        nb.add(f, text="  Export MP3  ")
        f.columnconfigure(1, weight=1)
        r = 0

        ttk.Checkbutton(f, text="Générer aussi un fichier MP3 (nécessite ffmpeg)",
                        variable=self.v_export_mp3).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=2, sticky="ew", pady=12)
        r += 1

        ttk.Label(f, text="Débit (bitrate)").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(f, textvariable=self.v_mp3_bitrate, values=BITRATES,
                     state="readonly", width=10).grid(row=r, column=1, sticky="e", pady=4)
        r += 1

        ttk.Checkbutton(f, text="Mono (recommandé pour le podcast)",
                        variable=self.v_mp3_mono).grid(row=r, column=0, columnspan=2, sticky="w", pady=4)
        r += 1

        ttk.Label(f, text="Source du MP3").grid(row=r, column=0, sticky="w", pady=4)
        cb = ttk.Combobox(f, textvariable=self.v_mp3_source,
                          values=["auto", "resample", "natif"], state="readonly", width=10)
        cb.grid(row=r, column=1, sticky="e", pady=4)
        Tooltip(cb, "auto = version rééchantillonnée si disponible, sinon natif 24 kHz.")
        r += 1

        ttk.Separator(f).grid(row=r, column=0, columnspan=2, sticky="ew", pady=12)
        r += 1
        ttk.Label(f, text="Chemin de ffmpeg (optionnel)").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(0, 2))
        r += 1
        cadre_path = ttk.Frame(f)
        cadre_path.grid(row=r, column=0, columnspan=2, sticky="ew")
        cadre_path.columnconfigure(0, weight=1)
        ef = ttk.Entry(cadre_path, textvariable=self.v_ffmpeg_path)
        ef.grid(row=0, column=0, sticky="ew")
        Tooltip(ef, "Laisse vide pour la détection automatique (PATH, winget,\n"
                    "Chocolatey, installation manuelle). Sinon, indique ffmpeg.exe\n"
                    "ou le dossier qui le contient.")
        ttk.Button(cadre_path, text="Parcourir…",
                   command=self._choisir_ffmpeg).grid(row=0, column=1, padx=(6, 0))
        r += 1
        ttk.Button(f, text="🔍 Détecter ffmpeg",
                   command=lambda: self._detecter_ffmpeg(silencieux=False)).grid(
            row=r, column=0, sticky="w", pady=(6, 0))
        r += 1
        ttk.Label(f, text="Inutile de modifier le PATH système : l'application sait "
                          "retrouver ffmpeg, ou tu peux pointer son chemin ici.",
                  style="Sub.TLabel", wraplength=450, justify="left").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _onglet_video(self, nb):
        """Montage d'une vidéo à partir d'une image fixe et d'une piste audio."""
        f = ttk.Frame(nb, padding=14)
        nb.add(f, text="  Vidéo  ")
        f.columnconfigure(1, weight=1)
        r = 0

        ttk.Label(f, text="Génère un fichier vidéo à partir d'une image fixe et "
                          "d'une piste audio (habillage de podcast, chapitre "
                          "audio, mise en ligne YouTube…).",
                  style="Sub.TLabel", wraplength=460, justify="left").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 10))
        r += 1

        # ------------------------------------------------------------- image
        ttk.Label(f, text="Image de fond (JPG, PNG, WEBP…)").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 2))
        r += 1
        ei = ttk.Entry(f, textvariable=self.v_video_image)
        ei.grid(row=r, column=0, columnspan=2, sticky="ew")
        Tooltip(ei, "Image affichée pendant toute la durée de la vidéo.\n"
                    "Idéalement au moins aussi grande que la résolution choisie.")
        ttk.Button(f, text="Parcourir…", command=self._choisir_image).grid(
            row=r, column=2, padx=(6, 0))
        r += 1

        # ------------------------------------------------------------- audio
        ttk.Label(f, text="Piste audio").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(12, 2))
        r += 1
        ea = ttk.Entry(f, textvariable=self.v_video_audio)
        ea.grid(row=r, column=0, columnspan=2, sticky="ew")
        Tooltip(ea, "WAV, MP3, FLAC… La vidéo s'arrête exactement à la fin de "
                    "cette piste.")
        ttk.Button(f, text="Parcourir…", command=self._choisir_audio_video).grid(
            row=r, column=2, padx=(6, 0))
        r += 1
        ttk.Button(f, text="↩ Utiliser le dernier rendu TTS",
                   command=self._utiliser_dernier_rendu).grid(
            row=r, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(f, text="Enchaîner automatiquement après la génération audio",
                        variable=self.v_video_auto).grid(
            row=r, column=1, columnspan=2, sticky="e", pady=(4, 0))
        r += 1

        # ------------------------------------------------------------ sortie
        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=14)
        r += 1
        ttk.Label(f, text="Fichier vidéo de sortie (emplacement et nom)").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 2))
        r += 1
        es = ttk.Entry(f, textvariable=self.v_video_sortie)
        es.grid(row=r, column=0, columnspan=2, sticky="ew")
        Tooltip(es, "Chemin complet du fichier à créer.\n"
                    "L'extension détermine le conteneur : .mp4, .mkv ou .mov.")
        ttk.Button(f, text="Enregistrer sous…",
                   command=self._choisir_video_sortie).grid(row=r, column=2, padx=(6, 0))
        r += 1

        # ----------------------------------------------------------- réglages
        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=14)
        r += 1

        ttk.Label(f, text="Résolution").grid(row=r, column=0, sticky="w", pady=4)
        cbr = ttk.Combobox(f, textvariable=self.v_video_resolution,
                           values=list(RESOLUTIONS.keys()), state="readonly")
        cbr.grid(row=r, column=1, columnspan=2, sticky="ew", pady=4)
        Tooltip(cbr, "L'image est redimensionnée sans déformation puis centrée "
                     "sur un fond uni si les proportions diffèrent.")
        r += 1

        ttk.Label(f, text="Images par seconde").grid(row=r, column=0, sticky="w", pady=4)
        cbf = ttk.Combobox(f, textvariable=self.v_video_fps, values=FPS_VIDEO,
                           state="readonly", width=10)
        cbf.grid(row=r, column=1, columnspan=2, sticky="e", pady=4)
        Tooltip(cbf, "L'image ne bouge pas : 1 à 5 img/s suffisent et divisent\n"
                     "fortement le temps d'encodage et le poids du fichier.\n"
                     "25 ou 30 img/s si la plateforme de destination l'exige.")
        r += 1

        ttk.Label(f, text="Qualité vidéo (CRF)").grid(row=r, column=0, sticky="w", pady=4)
        ec = ttk.Entry(f, textvariable=self.v_video_crf, width=10)
        ec.grid(row=r, column=1, columnspan=2, sticky="e", pady=4)
        Tooltip(ec, "Échelle inversée : 18 = très haute qualité, 23 = standard,\n"
                    "28 = fichier léger. Pour une image fixe, 20 est confortable.")
        r += 1

        ttk.Label(f, text="Preset d'encodage").grid(row=r, column=0, sticky="w", pady=4)
        cbp = ttk.Combobox(f, textvariable=self.v_video_preset, values=PRESETS_X264,
                           state="readonly", width=12)
        cbp.grid(row=r, column=1, columnspan=2, sticky="e", pady=4)
        Tooltip(cbp, "Compromis vitesse / compression. « medium » convient ;\n"
                     "« veryfast » accélère nettement les longues durées.")
        r += 1

        ttk.Label(f, text="Couleur de fond").grid(row=r, column=0, sticky="w", pady=4)
        ef = ttk.Entry(f, textvariable=self.v_video_fond, width=12)
        ef.grid(row=r, column=1, columnspan=2, sticky="e", pady=4)
        Tooltip(ef, "Utilisée pour les bandes si l'image n'a pas les proportions\n"
                    "du cadre. Nom ffmpeg (black, white, gray) ou code #RRGGBB.")
        r += 1

        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=12)
        r += 1

        cba = ttk.Checkbutton(f, text="Copier la piste audio sans réencodage",
                              variable=self.v_video_audio_copie,
                              command=self._maj_etat_audio_video)
        cba.grid(row=r, column=0, columnspan=3, sticky="w")
        Tooltip(cba, "Plus rapide et sans perte, mais le conteneur doit accepter\n"
                     "le format source (un WAV n'entre pas dans un MP4).\n"
                     "Dans le doute, laisse décoché : réencodage en AAC.")
        r += 1

        ttk.Label(f, text="Débit audio (AAC)").grid(row=r, column=0, sticky="w", pady=4)
        self.cb_video_bitrate = ttk.Combobox(
            f, textvariable=self.v_video_audio_bitrate, values=BITRATES,
            state="readonly", width=10)
        self.cb_video_bitrate.grid(row=r, column=1, columnspan=2, sticky="e", pady=4)
        r += 1

        # ------------------------------------------------------------ actions
        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=14)
        r += 1
        cadre = ttk.Frame(f)
        cadre.grid(row=r, column=0, columnspan=3, sticky="ew")
        self.btn_video = ttk.Button(cadre, text="🎬  Générer la vidéo",
                                    style="Accent.TButton", command=self._lancer_video)
        self.btn_video.pack(side="left")
        self.btn_video_lire = ttk.Button(cadre, text="▶ Lire la vidéo",
                                         command=self._lire_video, state="disabled")
        self.btn_video_lire.pack(side="left", padx=6)
        self.btn_video_dossier = ttk.Button(cadre, text="📂 Dossier",
                                            command=self._ouvrir_dossier_video,
                                            state="disabled")
        self.btn_video_dossier.pack(side="left")
        r += 1
        ttk.Label(f, text="Le montage n'utilise que ffmpeg : ni GPU ni modèle "
                          "n'est sollicité. Le bouton « Arrêter » de la barre du "
                          "bas interrompt aussi l'encodage.",
                  style="Sub.TLabel", wraplength=460, justify="left").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self._maj_etat_audio_video()

    def _onglet_fusion(self, nb):
        """Fusion de plusieurs fichiers audio et génération du chapitrage."""
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="  Fusion & Chapitres  ")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=3)      # la liste des segments
        f.rowconfigure(9, weight=2)      # le chapitrage
        r = 0

        ttk.Label(f, text="Assemble plusieurs fichiers audio dans l'ordre choisi "
                          "et produit le chapitrage correspondant, prêt à coller "
                          "dans une description YouTube.",
                  style="Sub.TLabel", wraplength=560, justify="left").grid(
            row=r, column=0, sticky="w", pady=(0, 8))
        r += 1

        # ------------------------------------------------- barre d'outils liste
        outils = ttk.Frame(f)
        outils.grid(row=r, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(outils, text="➕ Ajouter…",
                   command=self._ajouter_segments).pack(side="left")
        ttk.Button(outils, text="🗑 Retirer",
                   command=self._retirer_segment).pack(side="left", padx=4)
        ttk.Button(outils, text="▲", width=3,
                   command=lambda: self._deplacer_segment(-1)).pack(side="left")
        ttk.Button(outils, text="▼", width=3,
                   command=lambda: self._deplacer_segment(1)).pack(side="left", padx=(2, 4))
        ttk.Button(outils, text="✎ Renommer",
                   command=self._renommer_segment).pack(side="left")
        ttk.Button(outils, text="Tout vider",
                   command=self._vider_segments).pack(side="left", padx=4)
        ttk.Label(outils, textvariable=self.v_fusion_info,
                  style="Sub.TLabel").pack(side="right")
        r += 1

        # ------------------------------------------------------ liste ordonnée
        cadre_liste = ttk.Frame(f)
        cadre_liste.grid(row=r, column=0, sticky="nsew")
        cadre_liste.columnconfigure(0, weight=1)
        cadre_liste.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            cadre_liste, columns=("n", "libelle", "duree", "fichier"),
            show="headings", height=8, selectmode="browse")
        self.tree.heading("n", text="#")
        self.tree.heading("libelle", text="Libellé du chapitre")
        self.tree.heading("duree", text="Durée")
        self.tree.heading("fichier", text="Fichier")
        self.tree.column("n", width=34, anchor="center", stretch=False)
        self.tree.column("libelle", width=230, anchor="w")
        self.tree.column("duree", width=64, anchor="e", stretch=False)
        self.tree.column("fichier", width=200, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        defil = ttk.Scrollbar(cadre_liste, orient="vertical",
                              command=self.tree.yview)
        defil.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=defil.set)
        self.tree.bind("<Double-1>", self._editer_libelle_inline)
        Tooltip(self.tree, "Double-clique sur un libellé pour le modifier.\n"
                           "L'ordre de la liste est l'ordre du montage.")
        r += 1

        ttk.Label(f, text="Double-clic sur un libellé pour l'éditer • "
                          "les flèches ▲▼ réordonnent les segments",
                  style="Sub.TLabel").grid(row=r, column=0, sticky="w", pady=(3, 0))
        r += 1

        # ------------------------------------------------------------ sortie
        ttk.Separator(f).grid(row=r, column=0, sticky="ew", pady=10)
        r += 1
        cadre_sortie = ttk.Frame(f)
        cadre_sortie.grid(row=r, column=0, sticky="ew")
        cadre_sortie.columnconfigure(0, weight=1)
        ttk.Label(cadre_sortie, text="Fichier fusionné de sortie").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))
        ef = ttk.Entry(cadre_sortie, textvariable=self.v_fusion_sortie)
        ef.grid(row=1, column=0, sticky="ew")
        Tooltip(ef, "L'extension choisit le format : .mp3, .wav, .m4a, .flac, .opus.\n"
                    "Les chapitres intégrés exigent un conteneur compatible "
                    "(MP3, M4A, MKV, Opus).")
        ttk.Button(cadre_sortie, text="Enregistrer sous…",
                   command=self._choisir_fusion_sortie).grid(row=1, column=1, padx=(6, 0))
        r += 1

        # ---------------------------------------------------------- réglages
        reglages = ttk.Frame(f)
        reglages.grid(row=r, column=0, sticky="ew", pady=(10, 0))
        reglages.columnconfigure(1, weight=1)
        reglages.columnconfigure(3, weight=1)

        ttk.Label(reglages, text="Silence entre segments (s)").grid(
            row=0, column=0, sticky="w", pady=3)
        es = ttk.Entry(reglages, textvariable=self.v_fusion_silence, width=8)
        es.grid(row=0, column=1, sticky="w", padx=(6, 16), pady=3)
        Tooltip(es, "Blanc inséré entre deux segments. Il est compté dans le\n"
                    "chapitrage : l'horodatage pointe sur le début réel de l'audio.")

        ttk.Label(reglages, text="Fréquence (Hz)").grid(row=0, column=2, sticky="w", pady=3)
        ttk.Combobox(reglages, textvariable=self.v_fusion_sample_rate,
                     values=FREQUENCES_FUSION, state="readonly", width=8).grid(
            row=0, column=3, sticky="w", padx=6, pady=3)

        ttk.Label(reglages, text="Débit (si compressé)").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(reglages, textvariable=self.v_fusion_bitrate, values=BITRATES,
                     state="readonly", width=8).grid(
            row=1, column=1, sticky="w", padx=(6, 16), pady=3)
        ttk.Checkbutton(reglages, text="Mono", variable=self.v_fusion_mono).grid(
            row=1, column=2, columnspan=2, sticky="w", pady=3)

        ttk.Label(reglages, text="Normalisation").grid(row=2, column=0, sticky="w", pady=3)
        cbn = ttk.Combobox(reglages, textvariable=self.v_fusion_normalisation,
                           values=list(NORMALISATIONS.keys()), state="readonly")
        cbn.grid(row=2, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        Tooltip(cbn, "« Par segment » égalise des fichiers d'origines différentes\n"
                     "(voix clonée, intro musicale, extrait d'interview).\n"
                     "« Globale » préserve leurs écarts relatifs.")

        ttk.Label(reglages, text="Cible de loudness (LUFS)").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(reglages, textvariable=self.v_fusion_lufs, values=CIBLES_LUFS,
                     state="readonly").grid(row=3, column=1, columnspan=3,
                                            sticky="ew", padx=6, pady=3)

        ttk.Label(reglages, text="Format du chapitrage").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(reglages, textvariable=self.v_fusion_style,
                     values=list(STYLES_CHAPITRES.keys()), state="readonly").grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        r += 1

        options = ttk.Frame(f)
        options.grid(row=r, column=0, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(options, text="Enregistrer un fichier -chapitres.txt",
                        variable=self.v_fusion_txt).pack(side="left")
        cbi = ttk.Checkbutton(options, text="Intégrer les chapitres au fichier",
                              variable=self.v_fusion_integrer)
        cbi.pack(side="left", padx=12)
        Tooltip(cbi, "Écrit les chapitres dans les métadonnées (lecteurs de podcast).\n"
                     "Sans effet sur un WAV ou un FLAC, qui ne les gèrent pas.")
        r += 1

        # --------------------------------------------------------- chapitrage
        ttk.Separator(f).grid(row=r, column=0, sticky="ew", pady=10)
        r += 1
        barre_ch = ttk.Frame(f)
        barre_ch.grid(row=r, column=0, sticky="ew", pady=(0, 3))
        ttk.Label(barre_ch, text="Chapitrage généré",
                  style="TLabelframe.Label").pack(side="left")
        ttk.Button(barre_ch, text="📋 Copier",
                   command=self._copier_chapitres).pack(side="right")
        ttk.Button(barre_ch, text="💾 Enregistrer…",
                   command=self._enregistrer_chapitres).pack(side="right", padx=4)
        ttk.Button(barre_ch, text="🔄 Recalculer",
                   command=self._recalculer_chapitres).pack(side="right")
        r += 1

        self.txt_chapitres = scrolledtext.ScrolledText(
            f, height=7, wrap="none", font=("Consolas", 10),
            relief="solid", borderwidth=1, padx=8, pady=6)
        self.txt_chapitres.grid(row=r, column=0, sticky="nsew")
        r += 1

        # Le silence et le format d'affichage changent les horodatages :
        # le chapitrage se rafraîchit tout seul.
        for variable in (self.v_fusion_silence, self.v_fusion_style):
            variable.trace_add("write", lambda *_: self._rafraichir_chapitrage())
        self.v_fusion_sortie.trace_add("write", self._sortie_fusion_modifiee)

        # ------------------------------------------------------------ actions
        cadre_act = ttk.Frame(f)
        cadre_act.grid(row=r, column=0, sticky="ew", pady=(10, 0))
        self.btn_fusion = ttk.Button(cadre_act, text="🔗  Fusionner",
                                     style="Accent.TButton", command=self._lancer_fusion)
        self.btn_fusion.pack(side="left")
        self.btn_fusion_lire = ttk.Button(cadre_act, text="🔊 Lire",
                                          command=self._lire_fusion, state="disabled")
        self.btn_fusion_lire.pack(side="left", padx=6)
        self.btn_fusion_dossier = ttk.Button(cadre_act, text="📂 Dossier",
                                             command=self._ouvrir_dossier_fusion,
                                             state="disabled")
        self.btn_fusion_dossier.pack(side="left")
        self.btn_fusion_video = ttk.Button(cadre_act, text="→ Envoyer vers Vidéo",
                                           command=self._fusion_vers_video,
                                           state="disabled")
        self.btn_fusion_video.pack(side="left", padx=6)

    def _onglet_systeme(self, nb):
        f = ttk.Frame(nb, padding=14)
        nb.add(f, text="  Système  ")
        f.columnconfigure(1, weight=1)
        r = 0

        ttk.Label(f, text="Identifiant du modèle").grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 2))
        r += 1
        ttk.Entry(f, textvariable=self.v_model_id).grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1

        ttk.Label(f, text="Device").grid(row=r, column=0, sticky="w", pady=(12, 2))
        r += 1
        ttk.Combobox(f, textvariable=self.v_device, values=["auto", "cuda:0", "cpu"],
                     state="readonly").grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1

        ttk.Label(f, text="Token Hugging Face (optionnel)").grid(row=r, column=0, columnspan=2, sticky="w", pady=(12, 2))
        r += 1
        e = ttk.Entry(f, textvariable=self.v_hf_token, show="•")
        e.grid(row=r, column=0, columnspan=2, sticky="ew")
        Tooltip(e, "Laisse vide pour utiliser la variable d'environnement HF_TOKEN\n"
                   "ou le login CLI. Non enregistré par défaut, par sécurité.")
        r += 1
        ttk.Checkbutton(f, text="Mémoriser le token dans la configuration (déconseillé)",
                        variable=self.v_remember_token).grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 0))
        r += 1

        ttk.Separator(f).grid(row=r, column=0, columnspan=2, sticky="ew", pady=14)
        r += 1
        ttk.Button(f, text="🔍 Vérifier l'environnement",
                   command=self._verifier_env).grid(row=r, column=0, sticky="w")
        r += 1
        cadre = ttk.Frame(f)
        cadre.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Button(cadre, text="💾 Sauver config",
                   command=self._sauver_config_dialog).pack(side="left")
        ttk.Button(cadre, text="📁 Charger config",
                   command=self._charger_config_dialog).pack(side="left", padx=6)

    # ----------------------------------------------------------- sélecteurs
    def _choisir_ref(self):
        chemin = filedialog.askopenfilename(
            title="Choisir la voix de référence",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("Tous", "*.*")])
        if not chemin:
            return
        self.v_ref_audio.set(chemin)
        # Chargement automatique d'une transcription sidecar éventuelle
        base = os.path.splitext(chemin)[0]
        for cand in (base + ".txt", base + "-ref.txt", base + "-transcription.txt"):
            if os.path.exists(cand):
                try:
                    with open(cand, encoding="utf-8", errors="replace") as fp:
                        self.txt_ref.delete("1.0", "end")
                        self.txt_ref.insert("1.0", fp.read().strip())
                    self._log(f"Transcription de référence chargée : {os.path.basename(cand)}")
                except Exception:
                    pass
                break

    def _choisir_texte(self):
        chemin = filedialog.askopenfilename(
            title="Choisir le texte à lire",
            filetypes=[("Texte et Word", "*.txt *.md *.docx"),
                       ("Texte", "*.txt *.md"), ("Word", "*.docx"), ("Tous", "*.*")])
        if not chemin:
            return
        self.v_text_file.set(chemin)
        # Pré-remplissage du dossier et du nom de sortie
        if not self.v_output_dir.get().strip():
            self.v_output_dir.set(os.path.dirname(chemin))
        self.v_output_base.set(os.path.splitext(os.path.basename(chemin))[0])

    def _choisir_dossier(self):
        chemin = filedialog.askdirectory(title="Dossier de sortie")
        if chemin:
            self.v_output_dir.set(chemin)

    def _choisir_ffmpeg(self):
        chemin = filedialog.askopenfilename(
            title="Localiser ffmpeg",
            filetypes=[("ffmpeg", "ffmpeg.exe"), ("Tous les fichiers", "*.*")])
        if chemin:
            self.v_ffmpeg_path.set(chemin)
            self._detecter_ffmpeg(silencieux=False)

    def _detecter_ffmpeg(self, silencieux=True):
        ff = trouver_ffmpeg(self.v_ffmpeg_path.get().strip())
        if ff:
            if not self.v_ffmpeg_path.get().strip():
                self.v_ffmpeg_path.set(ff)      # mémorise l'emplacement trouvé
            self._log(f"ffmpeg détecté : {ff}")
            if not silencieux:
                messagebox.showinfo("ffmpeg", f"ffmpeg trouvé :\n{ff}")
        else:
            self._log("ffmpeg introuvable — installe-le ou indique son chemin.")
            if not silencieux:
                messagebox.showwarning(
                    "ffmpeg introuvable",
                    "ffmpeg n'a pas été trouvé.\n\n"
                    "Installe-le :\n    winget install --id=Gyan.FFmpeg -e\n\n"
                    "puis clique de nouveau sur « Détecter », ou indique "
                    "directement le chemin de ffmpeg.exe.")

    # --------------------------------------------------------- vidéo : choix
    def _choisir_image(self):
        chemin = filedialog.askopenfilename(
            title="Choisir l'image de fond",
            filetypes=[("Images", EXTENSIONS_IMAGE), ("Tous les fichiers", "*.*")])
        if not chemin:
            return
        self.v_video_image.set(chemin)
        self._proposer_sortie_video()

    def _choisir_audio_video(self):
        chemin = filedialog.askopenfilename(
            title="Choisir la piste audio de la vidéo",
            filetypes=[("Audio", EXTENSIONS_AUDIO), ("Tous les fichiers", "*.*")])
        if not chemin:
            return
        self.v_video_audio.set(chemin)
        self._proposer_sortie_video()

    def _utiliser_dernier_rendu(self):
        """Reprend le fichier produit par la dernière génération TTS."""
        if not self.dernier_fichier or not os.path.exists(self.dernier_fichier):
            messagebox.showinfo(
                "Aucun rendu",
                "Aucun rendu audio n'a encore été généré dans cette session.\n"
                "Lance une génération, ou sélectionne un fichier manuellement.")
            return
        self.v_video_audio.set(self.dernier_fichier)
        self._proposer_sortie_video()
        self._log(f"Piste audio de la vidéo : {os.path.basename(self.dernier_fichier)}")

    def _choisir_video_sortie(self):
        initial = self.v_video_sortie.get().strip()
        dossier = os.path.dirname(initial) if initial else (
            self.v_output_dir.get().strip() or os.path.expanduser("~"))
        nom = os.path.basename(initial) if initial else self._nom_video_propose()
        chemin = filedialog.asksaveasfilename(
            title="Enregistrer la vidéo sous…", defaultextension=".mp4",
            initialdir=dossier, initialfile=nom, filetypes=FORMATS_VIDEO)
        if chemin:
            self.v_video_sortie.set(chemin)

    def _nom_video_propose(self):
        """Nom de fichier suggéré, déduit de l'audio puis du nom de base TTS."""
        audio = self.v_video_audio.get().strip()
        if audio:
            return os.path.splitext(os.path.basename(audio))[0] + ".mp4"
        base = self.v_output_base.get().strip() or "video"
        return base + ".mp4"

    def _proposer_sortie_video(self):
        """Pré-remplit le chemin de sortie tant que l'utilisateur n'a rien
        choisi lui-même — sans jamais écraser une saisie manuelle."""
        if self.v_video_sortie.get().strip():
            return
        audio = self.v_video_audio.get().strip()
        image = self.v_video_image.get().strip()
        reference = audio or image
        if not reference:
            return
        dossier = self.v_output_dir.get().strip() or os.path.dirname(reference)
        self.v_video_sortie.set(os.path.join(dossier, self._nom_video_propose()))

    def _maj_etat_audio_video(self):
        """Le débit AAC n'a de sens que si l'audio est réencodé."""
        if hasattr(self, "cb_video_bitrate"):
            self.cb_video_bitrate.config(
                state="disabled" if self.v_video_audio_copie.get() else "readonly")

    # ----------------------------------------------------- vidéo : génération
    def _construire_config_video(self):
        return {
            "image": self.v_video_image.get().strip(),
            "audio": self.v_video_audio.get().strip(),
            "sortie": self.v_video_sortie.get().strip(),
            "resolution": RESOLUTIONS.get(self.v_video_resolution.get(), "source"),
            "resolution_label": self.v_video_resolution.get(),
            "fps": self._int(self.v_video_fps, 25),
            "crf": max(0, min(51, self._int(self.v_video_crf, 20))),
            "preset": self.v_video_preset.get() or "medium",
            "couleur_fond": self.v_video_fond.get().strip() or "black",
            "audio_copie": self.v_video_audio_copie.get(),
            "audio_bitrate": self.v_video_audio_bitrate.get(),
            "video_auto": self.v_video_auto.get(),
            "ffmpeg_path": self.v_ffmpeg_path.get().strip(),
        }

    def _lancer_video(self, silencieux=False):
        """Démarre le montage. `silencieux` = enchaînement automatique après
        une génération TTS : on n'affiche alors pas de boîtes de dialogue."""
        if self.worker_video and self.worker_video.is_alive():
            return
        if self.worker and self.worker.is_alive():
            if not silencieux:
                messagebox.showinfo(
                    "Génération en cours",
                    "Attends la fin de la génération audio avant de lancer le montage.")
            return

        config = self._construire_config_video()

        manquant = None
        if not config["image"] or not os.path.exists(config["image"]):
            manquant = "Sélectionne une image de fond valide."
        elif not config["audio"] or not os.path.exists(config["audio"]):
            manquant = "Sélectionne une piste audio valide."
        elif not config["sortie"]:
            manquant = "Indique l'emplacement et le nom du fichier vidéo de sortie."
        if manquant:
            if not silencieux:
                messagebox.showwarning("Montage vidéo", manquant)
            else:
                self._log(f"🎬 Montage automatique annulé : {manquant}")
            return

        if os.path.exists(config["sortie"]) and not silencieux:
            if not messagebox.askyesno(
                    "Fichier existant",
                    f"{os.path.basename(config['sortie'])} existe déjà.\n\n"
                    "L'écraser ?"):
                return

        self.stop_video.clear()
        self.taches.add("video")
        self._maj_boutons()
        self.btn_video_lire.config(state="disabled")
        self.btn_video_dossier.config(state="disabled")
        self.progress.config(mode="determinate", maximum=100, value=0)
        self.v_progress.set("Montage vidéo…")

        self.worker_video = threading.Thread(
            target=worker_video,
            args=(config, self.bus, self.stop_video),
            daemon=True)
        self.worker_video.start()

    def _fin_video(self):
        self.taches.discard("video")
        self._maj_boutons()

    def _lire_video(self):
        if self.derniere_video:
            self._ouvrir_chemin(self.derniere_video)

    def _ouvrir_dossier_video(self):
        if self.derniere_video:
            self._ouvrir_chemin(os.path.dirname(self.derniere_video))

    # -------------------------------------------------- fusion : liste
    def _ajouter_segments(self):
        chemins = filedialog.askopenfilenames(
            title="Ajouter des fichiers audio à fusionner",
            filetypes=[("Audio", EXTENSIONS_AUDIO), ("Tous les fichiers", "*.*")])
        if not chemins:
            return
        self._enregistrer_segments(chemins)

    def _enregistrer_segments(self, chemins):
        """Ajoute les fichiers à la liste en mesurant leur durée au passage."""
        ffmpeg = trouver_ffmpeg(self.v_ffmpeg_path.get().strip())
        ffprobe = trouver_ffprobe(ffmpeg)
        deja = {s["chemin"] for s in self.segments}
        ajoutes = 0
        self.v_fusion_info.set("Analyse…")
        self.update_idletasks()
        for chemin in chemins:
            if chemin in deja:
                self._log(f"Déjà dans la liste, ignoré : {os.path.basename(chemin)}")
                continue
            duree = duree_media(chemin, ffprobe)
            if duree is None:
                self._log(f"⚠️  Durée illisible : {os.path.basename(chemin)} "
                          "(sera remesurée au moment de la fusion)")
            self.segments.append({
                "chemin": chemin,
                "libelle": libelle_depuis_nom(chemin),
                "duree": duree,
            })
            deja.add(chemin)
            ajoutes += 1
        if ajoutes:
            self._log(f"{ajoutes} fichier(s) ajouté(s) à la fusion.")
        self._rafraichir_liste()
        self._proposer_sortie_fusion()
        self._recalculer_chapitres()

    def _rafraichir_liste(self):
        """Reconstruit l'affichage à partir de self.segments (source unique de
        vérité) et met à jour le résumé durée / nombre de fichiers."""
        selection = self.tree.selection()
        index_selection = self.tree.index(selection[0]) if selection else None
        self.tree.delete(*self.tree.get_children())
        total = 0.0
        inconnue = False
        for i, seg in enumerate(self.segments):
            duree = seg.get("duree")
            if duree is None:
                inconnue = True
                aff = "—"
            else:
                total += duree
                aff = formater_horodate(duree)
            self.tree.insert("", "end", values=(
                i + 1, seg["libelle"], aff, os.path.basename(seg["chemin"])))
        if not self.segments:
            self.v_fusion_info.set("Aucun fichier.")
        else:
            silence = self._float(self.v_fusion_silence, 1.0)
            total += silence * max(0, len(self.segments) - 1)
            self.v_fusion_info.set(
                f"{len(self.segments)} fichier(s) • "
                f"{'≈ ' if inconnue else ''}{formater_horodate(total, total >= 3600)}")
        enfants = self.tree.get_children()
        if index_selection is not None and enfants:
            cible = min(index_selection, len(enfants) - 1)
            self.tree.selection_set(enfants[cible])
            self.tree.focus(enfants[cible])

    def _index_selection(self):
        selection = self.tree.selection()
        return self.tree.index(selection[0]) if selection else None

    def _retirer_segment(self):
        i = self._index_selection()
        if i is None:
            messagebox.showinfo("Fusion", "Sélectionne d'abord un fichier dans la liste.")
            return
        retire = self.segments.pop(i)
        self._log(f"Retiré de la fusion : {os.path.basename(retire['chemin'])}")
        self._rafraichir_liste()
        self._recalculer_chapitres()

    def _deplacer_segment(self, sens):
        i = self._index_selection()
        if i is None:
            return
        j = i + sens
        if not (0 <= j < len(self.segments)):
            return
        self.segments[i], self.segments[j] = self.segments[j], self.segments[i]
        self._rafraichir_liste()
        enfants = self.tree.get_children()
        self.tree.selection_set(enfants[j])
        self.tree.focus(enfants[j])
        self._recalculer_chapitres()

    def _vider_segments(self):
        if not self.segments:
            return
        if not messagebox.askyesno("Fusion", "Vider entièrement la liste ?"):
            return
        self.segments.clear()
        self._rafraichir_liste()
        self._recalculer_chapitres()

    def _renommer_segment(self):
        """Renommage par boîte de dialogue (équivalent au double-clic)."""
        i = self._index_selection()
        if i is None:
            messagebox.showinfo("Fusion", "Sélectionne d'abord un fichier dans la liste.")
            return
        from tkinter import simpledialog
        nouveau = simpledialog.askstring(
            "Libellé du chapitre", "Intitulé affiché dans le chapitrage :",
            initialvalue=self.segments[i]["libelle"], parent=self)
        if nouveau is not None:
            self.segments[i]["libelle"] = nouveau.strip()
            self._rafraichir_liste()
            self._recalculer_chapitres()

    def _editer_libelle_inline(self, event):
        """Édition directe du libellé dans la liste, au double-clic."""
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        if self.tree.identify_column(event.x) != "#2":   # colonne « Libellé »
            return
        ligne = self.tree.identify_row(event.y)
        if not ligne:
            return
        index = self.tree.index(ligne)
        bbox = self.tree.bbox(ligne, "#2")
        if not bbox:
            return
        x, y, largeur, hauteur = bbox

        self._fermer_editeur_libelle()      # jamais deux champs simultanés
        champ = ttk.Entry(self.tree)
        self._editeur_libelle = champ
        champ.place(x=x, y=y, width=largeur, height=hauteur)
        champ.insert(0, self.segments[index]["libelle"])
        champ.select_range(0, "end")
        champ.focus_set()

        # Détruire un champ qui a le focus déclenche un <FocusOut> : sans ce
        # verrou, valider() serait rappelé sur un widget déjà détruit.
        termine = {"fait": False}

        def valider(_=None):
            if termine["fait"]:
                return
            termine["fait"] = True
            valeur = champ.get().strip()
            self._fermer_editeur_libelle()
            if valeur != self.segments[index]["libelle"]:
                self.segments[index]["libelle"] = valeur
                self._rafraichir_liste()
                self._recalculer_chapitres()

        def annuler(_=None):
            if termine["fait"]:
                return
            termine["fait"] = True
            self._fermer_editeur_libelle()

        champ.bind("<Return>", valider)
        champ.bind("<KP_Enter>", valider)
        champ.bind("<FocusOut>", valider)
        champ.bind("<Escape>", annuler)

    def _fermer_editeur_libelle(self):
        champ = getattr(self, "_editeur_libelle", None)
        self._editeur_libelle = None
        if champ is not None:
            try:
                champ.destroy()
            except Exception:
                pass

    # ------------------------------------------------ fusion : chapitrage
    def _rafraichir_chapitrage(self):
        """Appelé à chaque frappe dans le champ « silence » : une saisie
        intermédiaire (« 1. », champ vide) ne doit jamais faire remonter
        d'exception depuis la boucle Tk."""
        try:
            self._rafraichir_liste()
            self._recalculer_chapitres()
        except Exception:
            pass

    def _recalculer_chapitres(self):
        """Recalcule et réaffiche le chapitrage à partir de l'état courant.
        N'exige aucun encodage : c'est un aperçu instantané."""
        if not hasattr(self, "txt_chapitres"):
            return
        self.txt_chapitres.delete("1.0", "end")
        if not self.segments:
            return
        if any(s.get("duree") is None for s in self.segments):
            self.txt_chapitres.insert(
                "1.0", "Durée de certains fichiers illisible sans ffmpeg.\n"
                       "Le chapitrage exact sera calculé au lancement de la fusion.")
            return
        silence = self._float(self.v_fusion_silence, 1.0)
        chapitres, total = calculer_chapitres(self.segments, silence)
        style = STYLES_CHAPITRES.get(self.v_fusion_style.get(), "youtube")
        self.txt_chapitres.insert("1.0", formater_chapitres(chapitres, total, style))
        return chapitres, total

    def _copier_chapitres(self):
        texte = self.txt_chapitres.get("1.0", "end").strip()
        if not texte:
            messagebox.showinfo("Chapitres", "Le chapitrage est vide.")
            return
        self.clipboard_clear()
        self.clipboard_append(texte)
        self._log("Chapitrage copié dans le presse-papiers.")

    def _enregistrer_chapitres(self):
        texte = self.txt_chapitres.get("1.0", "end").strip()
        if not texte:
            messagebox.showinfo("Chapitres", "Le chapitrage est vide.")
            return
        sortie = self.v_fusion_sortie.get().strip()
        initial = (os.path.splitext(os.path.basename(sortie))[0] + "-chapitres.txt"
                   if sortie else "chapitres.txt")
        chemin = filedialog.asksaveasfilename(
            title="Enregistrer le chapitrage", defaultextension=".txt",
            initialdir=os.path.dirname(sortie) if sortie else None,
            initialfile=initial, filetypes=[("Texte", "*.txt"), ("Tous", "*.*")])
        if not chemin:
            return
        try:
            with open(chemin, "w", encoding="utf-8") as f:
                f.write(texte + "\n")
            self._log(f"Chapitrage enregistré : {chemin}")
        except Exception as e:
            messagebox.showerror("Chapitres", str(e))

    # ------------------------------------------------ fusion : génération
    def _choisir_fusion_sortie(self):
        initial = self.v_fusion_sortie.get().strip()
        if initial:
            dossier = os.path.dirname(initial)
        elif self.segments:
            dossier = os.path.dirname(os.path.abspath(self.segments[0]["chemin"]))
        else:
            dossier = self.v_output_dir.get().strip() or os.path.expanduser("~")
        nom = os.path.basename(initial) if initial else self._nom_fusion_propose()
        chemin = filedialog.asksaveasfilename(
            title="Enregistrer le fichier fusionné sous…", defaultextension=".mp3",
            initialdir=dossier, initialfile=nom, filetypes=FORMATS_FUSION)
        if chemin:
            self.v_fusion_sortie.set(chemin)

    def _nom_fusion_propose(self):
        """Nom suggéré pour le fichier fusionné.

        Il est déduit du dossier qui contient les segments, jamais du nom de
        base de l'onglet Entrées : celui-ci porte le titre du DERNIER texte
        généré en TTS, ce qui donnait au fichier fusionné le nom d'un seul de
        ses chapitres."""
        if self.segments:
            dossier = os.path.dirname(os.path.abspath(self.segments[0]["chemin"]))
            nom = os.path.basename(dossier)
            # Un dossier de rangement n'apprend rien sur l'œuvre : on remonte
            # d'un cran (…/Le Phénomène Humain/PARTIES → « Le Phénomène Humain »).
            if nom.lower() in DOSSIERS_GENERIQUES:
                parent = os.path.basename(os.path.dirname(dossier))
                if parent:
                    nom = parent
            if nom:
                return f"{nom}-complet.mp3"
        return "fusion-complet.mp3"

    def _proposer_sortie_fusion(self):
        """Suggère un chemin de sortie, et le maintient à jour tant que
        l'utilisateur n'a pas choisi lui-même un nom."""
        if not self.segments:
            return
        if self.v_fusion_sortie.get().strip() and not self._sortie_fusion_auto:
            return
        # Le fichier fusionné atterrit là où vivent ses sources, pas dans le
        # dossier de sortie du TTS, qui peut désigner un tout autre projet.
        dossier = (os.path.dirname(os.path.abspath(self.segments[0]["chemin"]))
                   or self.v_output_dir.get().strip())
        self._definir_sortie_auto(os.path.join(dossier, self._nom_fusion_propose()))

    def _definir_sortie_auto(self, chemin):
        """Renseigne le champ sans que cela compte comme une saisie manuelle."""
        self._sortie_auto_en_cours = True
        try:
            self.v_fusion_sortie.set(chemin)
        finally:
            self._sortie_auto_en_cours = False
        self._sortie_fusion_auto = True

    def _sortie_fusion_modifiee(self, *_):
        """Dès que le champ est modifié à la main (ou via « Enregistrer sous… »),
        la proposition automatique cesse de l'écraser."""
        if not self._sortie_auto_en_cours:
            self._sortie_fusion_auto = False

    def _construire_config_fusion(self):
        lufs = self.v_fusion_lufs.get().split()[0]
        try:
            lufs = float(lufs)
        except ValueError:
            lufs = -16.0
        return {
            # copie défensive : le thread ne doit pas voir la liste bouger
            "segments": [dict(s) for s in self.segments],
            "sortie": self.v_fusion_sortie.get().strip(),
            "silence": max(0.0, self._float(self.v_fusion_silence, 1.0)),
            "sample_rate": self.v_fusion_sample_rate.get() or "44100",
            "mono": self.v_fusion_mono.get(),
            "bitrate": self.v_fusion_bitrate.get() or "192k",
            "normalisation": NORMALISATIONS.get(
                self.v_fusion_normalisation.get(), "aucune"),
            "lufs": lufs,
            "style_chapitres": STYLES_CHAPITRES.get(self.v_fusion_style.get(), "youtube"),
            "ecrire_txt": self.v_fusion_txt.get(),
            "integrer_chapitres": self.v_fusion_integrer.get(),
            "ffmpeg_path": self.v_ffmpeg_path.get().strip(),
        }

    def _config_fusion_persistable(self):
        """Version de la configuration destinée au JSON : on conserve les
        libellés saisis, mais pas les durées mesurées (recalculées à l'ouverture
        des fichiers)."""
        d = self._construire_config_fusion()
        d["segments"] = [{"chemin": s["chemin"], "libelle": s["libelle"]}
                         for s in self.segments]
        d.pop("ffmpeg_path", None)
        d["normalisation_label"] = self.v_fusion_normalisation.get()
        d["lufs_label"] = self.v_fusion_lufs.get()
        d["style_label"] = self.v_fusion_style.get()
        return d

    def _lancer_fusion(self):
        if self.taches:
            messagebox.showinfo(
                "Traitement en cours",
                "Attends la fin du traitement en cours avant de lancer la fusion.")
            return

        config = self._construire_config_fusion()
        if len(config["segments"]) < 2:
            messagebox.showwarning(
                "Fusion", "Ajoute au moins deux fichiers audio à fusionner.")
            return
        if not config["sortie"]:
            messagebox.showwarning(
                "Fusion", "Indique l'emplacement et le nom du fichier fusionné.")
            return
        sources = {os.path.abspath(s["chemin"]) for s in config["segments"]}
        if os.path.abspath(config["sortie"]) in sources:
            messagebox.showwarning(
                "Fusion", "Le fichier de sortie est aussi l'un des fichiers "
                          "sources. Choisis un autre nom.")
            return
        if os.path.exists(config["sortie"]):
            if not messagebox.askyesno(
                    "Fichier existant",
                    f"{os.path.basename(config['sortie'])} existe déjà.\n\nL'écraser ?"):
                return

        self.stop_fusion.clear()
        self.taches.add("fusion")
        self._maj_boutons()
        self.btn_fusion_lire.config(state="disabled")
        self.btn_fusion_dossier.config(state="disabled")
        self.btn_fusion_video.config(state="disabled")
        self.progress.config(mode="determinate", maximum=100, value=0)
        self.v_progress.set("Fusion audio…")

        self.worker_fusion = threading.Thread(
            target=worker_fusion, args=(config, self.bus, self.stop_fusion),
            daemon=True)
        self.worker_fusion.start()

    def _fin_fusion(self):
        self.taches.discard("fusion")
        self._maj_boutons()

    def _lire_fusion(self):
        if self.dernier_fusionne:
            self._ouvrir_chemin(self.dernier_fusionne)

    def _ouvrir_dossier_fusion(self):
        if self.dernier_fusionne:
            self._ouvrir_chemin(os.path.dirname(self.dernier_fusionne))

    def _fusion_vers_video(self):
        """Envoie le fichier fusionné dans l'onglet Vidéo comme piste audio."""
        if not self.dernier_fusionne:
            return
        self.v_video_audio.set(self.dernier_fusionne)
        self.v_video_sortie.set("")          # laisse proposer un nom cohérent
        self._proposer_sortie_video()
        self._log(f"Piste audio de la vidéo : {os.path.basename(self.dernier_fusionne)}")
        messagebox.showinfo(
            "Vidéo", "Le fichier fusionné est maintenant la piste audio de "
                     "l'onglet Vidéo.")

    # ------------------------------------------------------------- aperçu
    def _apercu_decoupage(self):
        chemin = self.v_text_file.get().strip()
        if not chemin or not os.path.exists(chemin):
            messagebox.showwarning("Aperçu", "Sélectionne d'abord un fichier texte valide.")
            return
        try:
            plan = construire_plan(lire_paragraphes(chemin), self._int(self.v_max_chars, 350),
                                   par_chapitres=self.v_livre.get())
        except Exception as e:
            messagebox.showerror("Aperçu", f"Erreur de lecture :\n{e}")
            return
        stats = plan["stats"]
        self._log("─" * 50)
        self._log(f"APERÇU — {os.path.basename(chemin)}")
        self._log(f"Chunks : {stats['nb_chunks']}  |  "
                  f"min {stats['min']} / max {stats['max']} / moy {stats['moyenne']} car.")
        self._log(f"Fins de paragraphe : {stats['nb_fin_para']}")
        if self.v_livre.get():
            self._log(f"Chapitres ({stats['nb_chapitres']}) : " + " | ".join(plan["titres"][:12])
                      + (" …" if len(plan["titres"]) > 12 else ""))
        if stats["voix"]:
            dossier = self.v_dossier_voix.get().strip()
            for nom in stats["voix"]:
                n = sum(1 for v in plan["voix"] if v == nom)
                trouve = trouver_voix(dossier, nom)
                etat = (f"✓ {os.path.basename(trouve[0])}" if trouve
                        else "✗ introuvable : la voix principale la remplacera")
                self._log(f"Voix [{nom}] : {n} morceau(x) — {etat}")
        extrait = stats["texte_nettoye"][:600]
        self._log("Texte nettoyé (extrait) :")
        self._log(extrait + ("…" if len(stats["texte_nettoye"]) > 600 else ""))
        self._log("─" * 50)

    def _verifier_env(self):
        chemin_ff = self.v_ffmpeg_path.get().strip()

        def tache():
            self.bus.log("─" * 50)
            self.bus.log("Vérification de l'environnement…")
            try:
                import torch
                self.bus.log(f"  torch {torch.__version__} — OK")
                if torch.cuda.is_available():
                    self.bus.log(f"  CUDA disponible — {torch.cuda.get_device_name(0)}")
                    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    self.bus.log(f"  VRAM totale : {vram:.1f} Go")
                else:
                    self.bus.log("  CUDA indisponible — exécution CPU.")
            except Exception as e:
                self.bus.log(f"  torch : ABSENT ou erreur ({e})")
            for mod, lib in [("qwen_tts", "qwen_tts"), ("soundfile", "soundfile"),
                             ("scipy", "scipy"), ("docx", "python-docx")]:
                try:
                    __import__(mod)
                    self.bus.log(f"  {lib} — OK")
                except Exception:
                    self.bus.log(f"  {lib} — ABSENT  (pip install {lib})")
            ff = trouver_ffmpeg(chemin_ff)
            self.bus.log("  ffmpeg — " + (f"OK ({ff})" if ff
                         else "ABSENT (installe-le ou indique son chemin, onglet Export)"))
            self.bus.log("Vérification terminée.")
            self.bus.log("─" * 50)
        threading.Thread(target=tache, daemon=True).start()

    # --------------------------------------------------------- génération
    def _precharger_bibliotheques(self):
        """Importe torch / qwen_tts / numpy / soundfile / scipy sur le THREAD
        PRINCIPAL, une fois la fenêtre affichée. Indispensable sous Windows :
        onnxruntime (tiré par qwen_tts) échoue si son premier import a lieu sur
        un thread secondaire."""
        self.v_progress.set("Chargement des bibliothèques (torch, qwen_tts)…")
        self.lbl_modele.config(text="● Chargement des bibliothèques…",
                               foreground="#a0852a")
        self._log("Chargement des bibliothèques natives sur le thread principal…")
        self._log("(torch, qwen_tts, onnxruntime — peut prendre quelques secondes)")
        self.update_idletasks()           # force l'affichage avant le blocage
        try:
            precharger_modules()
        except Exception as e:
            import traceback
            self._log(traceback.format_exc())
            self._log(f"❌ Échec du chargement des bibliothèques : {e}")
            self.v_progress.set("Bibliothèques indisponibles.")
            self.lbl_modele.config(text="● Bibliothèques indisponibles",
                                   foreground="#a05a5a")
            messagebox.showerror(
                "Bibliothèques",
                "Impossible de charger torch / qwen_tts :\n\n"
                f"{e}\n\n"
                "Vérifie que l'environnement qwen_env est bien actif.")
            return
        self.pret = True
        self._maj_boutons()
        self.v_progress.set("Prêt.")
        self.lbl_modele.config(text="● Bibliothèques prêtes (modèle non chargé)",
                               foreground="#6a6a85")
        self._log("Bibliothèques chargées — prêt à générer.")
        self._detecter_ffmpeg(silencieux=True)   # localise ffmpeg sans déranger

    def _construire_config(self):
        freq = self.v_freq_supp.get()
        freq_val = None if freq in ("Aucune", "", None) else int(freq)
        return {
            "ref_audio": self.v_ref_audio.get().strip(),
            "text_file": self.v_text_file.get().strip(),
            "ref_text": self.txt_ref.get("1.0", "end").strip(),
            "instruction": self.v_instruction.get().strip(),
            "language": self.v_language.get(),
            "output_dir": self.v_output_dir.get().strip(),
            "output_base": self.v_output_base.get().strip(),
            "horodater": self.v_horodater.get(),
            "max_chars": self._int(self.v_max_chars, 350),
            "taille_lot": max(1, self._int(self.v_taille_lot, TAILLE_LOT)),
            "qc": self.v_qc.get(),
            "srt": self.v_srt.get(),
            "dossier_voix": self.v_dossier_voix.get().strip(),
            "livre": self.v_livre.get(),
            "livre_titre": self.v_livre_titre.get().strip(),
            "livre_auteur": self.v_livre_auteur.get().strip(),
            "livre_couverture": self.v_livre_couverture.get().strip(),
            "garder_morceaux": self.v_garder_morceaux.get(),
            "sil_virgule": self._float(self.v_sil_virgule, 0.15),
            "sil_point": self._float(self.v_sil_point, 0.40),
            "sil_para": self._float(self.v_sil_para, 0.70),
            "target_peak": self._float(self.v_target_peak, -3.0),
            "freq_supp": freq_val,
            "export_mp3": self.v_export_mp3.get(),
            "mp3_bitrate": self.v_mp3_bitrate.get(),
            "mp3_mono": self.v_mp3_mono.get(),
            "mp3_source": self.v_mp3_source.get(),
            "ffmpeg_path": self.v_ffmpeg_path.get().strip(),
            "model_id": self.v_model_id.get().strip(),
            "device": self.v_device.get(),
            "hf_token": self.v_hf_token.get().strip(),
        }

    def _lancer(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.pret:
            messagebox.showinfo("Patiente",
                                "Les bibliothèques sont encore en cours de chargement.")
            return
        config = self._construire_config()
        # Validations
        if not config["ref_audio"] or not os.path.exists(config["ref_audio"]):
            messagebox.showwarning("Voix manquante", "Sélectionne une voix de référence valide.")
            return
        if not config["text_file"] or not os.path.exists(config["text_file"]):
            messagebox.showwarning("Texte manquant", "Sélectionne un fichier texte valide.")
            return
        if not config["ref_text"]:
            if not messagebox.askyesno(
                    "Transcription vide",
                    "La transcription de la voix de référence est vide.\n"
                    "Le clonage sera de moins bonne qualité.\n\nContinuer quand même ?"):
                return

        if config["livre"] and config["livre_couverture"] and not os.path.isfile(config["livre_couverture"]):
            messagebox.showwarning("Couverture introuvable",
                                   "L'image de couverture n'existe pas : le livre sera créé sans couverture.")
        if config["dossier_voix"] and not os.path.isdir(config["dossier_voix"]):
            messagebox.showwarning("Dossier des voix introuvable",
                                   "Le dossier des voix n'existe pas : toutes les répliques "
                                   "seront lues avec la voix principale.")

        self.stop_event.clear()
        self.taches.add("tts")
        self._maj_boutons()
        self.btn_ouvrir.config(state="disabled")
        self.btn_lire.config(state="disabled")
        self.progress.config(value=0, maximum=100)
        self.t_debut = time.time()
        self._tic_chrono()

        self.worker = threading.Thread(
            target=worker_generation,
            args=(config, self.bus, self.stop_event, self.model_cache),
            daemon=True)
        self.worker.start()

    def _arreter(self):
        self.stop_event.set()
        self.stop_video.set()          # interrompt aussi un encodage ffmpeg
        self.stop_fusion.set()
        self.v_progress.set("Arrêt en cours…")
        self.btn_stop.config(state="disabled")

    def _tic_chrono(self):
        if self.t_debut and self.worker and self.worker.is_alive():
            ecoule = int(time.time() - self.t_debut)
            base = self.v_progress.get().split("  •")[0]
            self.v_progress.set(f"{base}  •  {ecoule // 60}m {ecoule % 60:02d}s")
            self.after(1000, self._tic_chrono)

    # ------------------------------------------------------- file de messages
    def _pomper_messages(self):
        try:
            while True:
                msg = self.file_msgs.get_nowait()
                self._traiter_message(msg)
        except queue.Empty:
            pass
        self.after(80, self._pomper_messages)

    def _traiter_message(self, msg):
        type_ = msg[0]
        if type_ == "log":
            self._log(msg[1])
        elif type_ == "status":
            self.v_progress.set(msg[1])
        elif type_ == "progress":
            i, n = msg[1], msg[2]
            self.progress.config(maximum=n, value=i)
            self.v_progress.set(f"Chunk {i}/{n}")
        elif type_ == "vram":
            self._log(f"         💾 VRAM utilisée : {msg[1]:.1f} Go")
        elif type_ == "model":
            self.lbl_modele.config(text="● Modèle chargé", foreground="#3a8a4a")
        elif type_ == "error":
            self._fin_generation()
            self._log(f"❌ ERREUR : {msg[1]}")
            self.v_progress.set("Erreur.")
            messagebox.showerror("Erreur", msg[1])
        elif type_ == "transcription":
            texte, duree = msg[1], msg[2]
            self.taches.discard("transcription")
            self._maj_boutons()
            self.txt_ref.delete("1.0", "end")
            self.txt_ref.insert("1.0", texte)
            self.v_progress.set("Transcription terminée.")
            self._log(f"   ✓ Transcription ({duree:.0f} s d'audio) : relisez-la et corrigez si besoin.")
            if duree > 20:
                self._log("   💡 Un extrait de 10 à 15 s suffit au clonage et laisse "
                          "plus de mémoire pour de grands lots.")
        elif type_ == "transcription_erreur":
            self.taches.discard("transcription")
            self._maj_boutons()
            self.v_progress.set("Transcription impossible.")
            self._log(f"❌ Transcription impossible : {msg[1]}")
            messagebox.showerror("Transcription", msg[1])
        elif type_ == "done":
            d = msg[1]
            self._fin_generation()
            self.dernier_projet = d.get("projet") or self.dernier_projet
            self.dernier_dossier = d["dossier"]
            self.dernier_fichier = d["principal"]
            self.btn_ouvrir.config(state="normal")
            self.btn_lire.config(state="normal")
            mn, sc = int(d["duree_audio"] // 60), int(d["duree_audio"] % 60)
            calc = int(d["duree_calcul"])
            etat = "Rendu partiel" if d["partiel"] else "Terminé"
            self.v_progress.set(f"{etat} — {mn}m {sc:02d}s d'audio")
            self._log("═" * 50)
            self._log(f"✅ {etat} : {d['total'] - d['echecs']}/{d['total']} chunks "
                      f"en {calc // 60}m {calc % 60:02d}s")
            self._log(f"   Audio : {mn} min {sc} sec  →  {d['principal']}")
            self._preparer_video_apres_audio(d)

        # ------------------------------------------------------------ vidéo
        elif type_ == "vstatus":
            self.v_progress.set(msg[1])
        elif type_ == "vprogress":
            self.progress.config(maximum=100, value=msg[1])
        elif type_ == "verror":
            self._fin_video()
            self._log(f"❌ ERREUR VIDÉO : {msg[1]}")
            self.v_progress.set("Montage interrompu.")
            messagebox.showerror("Montage vidéo", msg[1])
        # ----------------------------------------------------------- fusion
        elif type_ == "fstatus":
            self.v_progress.set(msg[1])
        elif type_ == "fprogress":
            self.progress.config(maximum=100, value=msg[1])
        elif type_ == "fchapitres":
            self.txt_chapitres.delete("1.0", "end")
            self.txt_chapitres.insert("1.0", msg[1])
        elif type_ == "ferror":
            self._fin_fusion()
            self._log(f"❌ ERREUR FUSION : {msg[1]}")
            self.v_progress.set("Fusion interrompue.")
            messagebox.showerror("Fusion audio", msg[1])
        elif type_ == "fdone":
            d = msg[1]
            self._fin_fusion()
            self.dernier_fusionne = d["fichier"]
            self.btn_fusion_lire.config(state="normal")
            self.btn_fusion_dossier.config(state="normal")
            self.btn_fusion_video.config(state="normal")
            self.progress.config(value=100)
            duree = d["duree_totale"]
            self.v_progress.set(
                f"Fusion terminée — {int(duree // 60)}m {int(duree % 60):02d}s")
            self._log("═" * 50)
            self._log(f"✅ {d['nb_segments']} fichiers fusionnés en "
                      f"{int(d['duree_calcul'] // 60)}m "
                      f"{int(d['duree_calcul'] % 60):02d}s")
            self._log(f"   {d['fichier']}  ({d['taille_mo']:.1f} Mo)")
            ecrite = d.get("duree_ecrite")
            if ecrite is not None and abs(ecrite - duree) > max(2.0, duree * 0.01):
                self.v_progress.set("⚠️ Fichier plus court que prévu — voir le journal.")
                messagebox.showwarning(
                    "Fusion incomplète",
                    f"Le fichier écrit dure {formater_horodate(ecrite, True)} "
                    f"au lieu de {formater_horodate(duree, True)}.\n\n"
                    "Le journal détaille les causes possibles. Si la destination "
                    "est un dossier synchronisé (Google Drive, OneDrive, Dropbox), "
                    "refais la fusion vers un disque local puis copie le résultat.")

        elif type_ == "vdone":
            d = msg[1]
            self._fin_video()
            self.derniere_video = d["fichier"]
            self.btn_video_lire.config(state="normal")
            self.btn_video_dossier.config(state="normal")
            self.progress.config(value=100)
            self.v_progress.set(f"Vidéo prête — {d['taille_mo']:.1f} Mo")
            self._log("═" * 50)
            self._log(f"🎬 Vidéo générée : {d['fichier']}")

    def _preparer_video_apres_audio(self, d):
        """À la fin d'un rendu TTS : propose le fichier produit comme piste
        audio du montage, et enchaîne si l'utilisateur l'a demandé."""
        fichiers = d.get("fichiers") or []
        mp3 = next((f for f in fichiers if f.lower().endswith(".mp3")), None)
        candidat = mp3 or d.get("principal")
        if not candidat:
            return
        if not self.v_video_audio.get().strip() or self.v_video_auto.get():
            self.v_video_audio.set(candidat)
            self._proposer_sortie_video()
        if self.v_video_auto.get() and not d.get("partiel"):
            self._log("🎬 Enchaînement automatique sur le montage vidéo…")
            self.after(400, lambda: self._lancer_video(silencieux=True))

    def _fin_generation(self):
        self.taches.discard("tts")
        self._maj_boutons()
        self.t_debut = None

    def _maj_boutons(self):
        """Un seul traitement lourd à la fois : les trois moteurs (TTS, montage
        vidéo, fusion) partagent la barre de progression et le bouton d'arrêt.
        L'état est déduit de self.taches, jamais de is_alive() — un thread peut
        rester « vivant » une fraction de seconde après son dernier message."""
        occupe = bool(self.taches)
        etat = "disabled" if occupe else "normal"
        self.btn_video.config(state=etat)
        self.btn_fusion.config(state=etat)
        self.btn_generer.config(state="normal" if (self.pret and not occupe) else "disabled")
        for nom in ("btn_transcrire", "btn_corriger"):
            if hasattr(self, nom):
                getattr(self, nom).config(state=etat)
        self.btn_stop.config(state="normal" if occupe else "disabled")

    # ---------------------------------------------------------------- sortie
    def _ouvrir_dossier(self):
        if not self.dernier_dossier:
            return
        self._ouvrir_chemin(self.dernier_dossier)

    def _lire_rendu(self):
        if not self.dernier_fichier:
            return
        self._ouvrir_chemin(self.dernier_fichier)

    def _ouvrir_chemin(self, chemin):
        try:
            if os.name == "nt":
                os.startfile(chemin)  # noqa
            elif shutil.which("open"):
                subprocess.Popen(["open", chemin])
            else:
                subprocess.Popen(["xdg-open", chemin])
        except Exception as e:
            messagebox.showerror("Ouverture", str(e))

    # ----------------------------------------------------------- config json
    def _sauver_config_dialog(self):
        chemin = filedialog.asksaveasfilename(
            title="Enregistrer la configuration", defaultextension=".json",
            initialfile="config_voix.json", filetypes=[("JSON", "*.json")])
        if chemin:
            self._sauver_config(chemin)
            self._log(f"Configuration enregistrée : {chemin}")

    def _charger_config_dialog(self):
        chemin = filedialog.askopenfilename(
            title="Charger une configuration", filetypes=[("JSON", "*.json")])
        if chemin:
            self._charger_config(chemin)
            self._log(f"Configuration chargée : {chemin}")

    def _config_dict(self):
        d = self._construire_config()
        video = self._construire_config_video()
        video.pop("ffmpeg_path", None)     # déjà présent dans la config audio
        d["video"] = video
        d["fusion"] = self._config_fusion_persistable()
        d["ref_text"] = self.txt_ref.get("1.0", "end").strip()
        d["freq_supp_raw"] = self.v_freq_supp.get()
        if not self.v_remember_token.get():
            d["hf_token"] = ""        # on n'enregistre jamais le token par défaut
        d["remember_token"] = self.v_remember_token.get()
        return d

    def _sauver_config(self, chemin):
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self._config_dict(), f, ensure_ascii=False, indent=2)

    def _charger_config(self, chemin, silencieux=False):
        with open(chemin, encoding="utf-8") as f:
            d = json.load(f)
        self.v_ref_audio.set(d.get("ref_audio", ""))
        self.v_text_file.set(d.get("text_file", ""))
        ref = d.get("ref_text", "").strip()
        if ref:                       # on préserve le texte par défaut si vide
            self.txt_ref.delete("1.0", "end")
            self.txt_ref.insert("1.0", ref)
        self.v_instruction.set(d.get("instruction", PRESETS_EMOTION[0]))
        self.v_language.set(d.get("language", "French"))
        self.v_output_dir.set(d.get("output_dir", ""))
        self.v_output_base.set(d.get("output_base", "rendu"))
        self.v_horodater.set(d.get("horodater", False))
        self.v_max_chars.set(d.get("max_chars", 350))
        self.v_taille_lot.set(TAILLE_LOT)      # la constante en tête de fichier prime
        self.v_qc.set(d.get("qc", False))
        self.v_srt.set(d.get("srt", True))
        self.v_dossier_voix.set(d.get("dossier_voix", ""))
        self.v_livre.set(d.get("livre", False))
        self.v_livre_titre.set(d.get("livre_titre", ""))
        self.v_livre_auteur.set(d.get("livre_auteur", ""))
        self.v_livre_couverture.set(d.get("livre_couverture", ""))
        self.v_garder_morceaux.set(d.get("garder_morceaux", True))
        self.v_sil_virgule.set(d.get("sil_virgule", 0.15))
        self.v_sil_point.set(d.get("sil_point", 0.40))
        self.v_sil_para.set(d.get("sil_para", 0.70))
        self.v_target_peak.set(d.get("target_peak", -3.0))
        self.v_freq_supp.set(d.get("freq_supp_raw", "44100"))
        self.v_export_mp3.set(d.get("export_mp3", True))
        self.v_mp3_bitrate.set(d.get("mp3_bitrate", "192k"))
        self.v_mp3_mono.set(d.get("mp3_mono", True))
        self.v_mp3_source.set(d.get("mp3_source", "auto"))
        self.v_ffmpeg_path.set(d.get("ffmpeg_path", ""))
        self.v_model_id.set(d.get("model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
        self.v_device.set(d.get("device", "auto"))
        self.v_remember_token.set(d.get("remember_token", False))
        if d.get("hf_token"):
            self.v_hf_token.set(d["hf_token"])

        # --- réglages du montage vidéo (absents des anciennes configs) ---
        v = d.get("video") or {}
        self.v_video_image.set(v.get("image", ""))
        self.v_video_audio.set(v.get("audio", ""))
        self.v_video_sortie.set(v.get("sortie", ""))
        self.v_video_resolution.set(
            v.get("resolution_label", "1920 × 1080 — 16:9 (YouTube)"))
        self.v_video_fps.set(str(v.get("fps", 25)))
        self.v_video_crf.set(v.get("crf", 20))
        self.v_video_preset.set(v.get("preset", "medium"))
        self.v_video_fond.set(v.get("couleur_fond", "black"))
        self.v_video_audio_copie.set(v.get("audio_copie", False))
        self.v_video_audio_bitrate.set(v.get("audio_bitrate", "192k"))
        self.v_video_auto.set(v.get("video_auto", False))
        self._maj_etat_audio_video()

        # --- réglages de la fusion audio ---
        u = d.get("fusion") or {}
        self.v_fusion_sortie.set(u.get("sortie", ""))
        self.v_fusion_silence.set(u.get("silence", 1.0))
        self.v_fusion_sample_rate.set(str(u.get("sample_rate", "44100")))
        self.v_fusion_mono.set(u.get("mono", True))
        self.v_fusion_bitrate.set(u.get("bitrate", "192k"))
        self.v_fusion_normalisation.set(u.get(
            "normalisation_label", "Par segment — harmonise les fichiers entre eux"))
        self.v_fusion_lufs.set(u.get("lufs_label", "-16 (podcast, Apple)"))
        self.v_fusion_style.set(u.get("style_label", "YouTube — 00:00 Libellé"))
        self.v_fusion_txt.set(u.get("ecrire_txt", True))
        self.v_fusion_integrer.set(u.get("integrer_chapitres", True))
        # Les fichiers disparus depuis la dernière session sont écartés, sinon
        # la liste renverrait à des chemins morts.
        self.segments = [
            {"chemin": s["chemin"], "libelle": s.get("libelle", ""), "duree": None}
            for s in u.get("segments", [])
            if isinstance(s, dict) and s.get("chemin") and os.path.exists(s["chemin"])
        ]
        if self.segments:
            ffprobe = trouver_ffprobe(trouver_ffmpeg(self.v_ffmpeg_path.get().strip()))
            for seg in self.segments:
                seg["duree"] = duree_media(seg["chemin"], ffprobe)
        self._rafraichir_liste()
        self._recalculer_chapitres()
        if not self.v_fusion_sortie.get().strip():
            self._sortie_fusion_auto = True
            self._proposer_sortie_fusion()

    # ---------------------------------------------------------------- logs
    def _log_init(self, m):
        self.log.configure(state="normal")
        self.log.insert("end", m + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log(self, m):
        self.log.configure(state="normal")
        self.log.insert("end", m + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _effacer_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _int(var, defaut):
        try:
            return int(float(var.get()))
        except Exception:
            return defaut

    @staticmethod
    def _float(var, defaut):
        try:
            return float(var.get())
        except Exception:
            return defaut

    def _fermer(self):
        if self.taches:
            if not messagebox.askyesno("Quitter",
                                       "Un traitement est en cours. Quitter quand même ?"):
                return
            self.stop_event.set()
            self.stop_video.set()
            self.stop_fusion.set()
        try:
            self._sauver_config(CONFIG_DEFAUT)
        except Exception:
            pass
        self.destroy()


def main():
    app = TTSStudio()
    app.mainloop()


if __name__ == "__main__":
    main()