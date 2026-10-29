# -*- coding: utf-8 -*-
# Copyright 2026 Denis Favre
# SPDX-License-Identifier: Apache-2.0
"""
================================================================================
  DIAGNOSTIC DE VITESSE — Qwen3-TTS   (v2 : avec CUDA Graph)
================================================================================
  Compare, sur TON GPU, trois façons de faire tourner le sous-modèle :
      origine  — transformers.generate() (bibliothèque d'origine)
      directe  — boucle directe (même résultat au bit près)
      graphe   — CUDA Graph, contrôlé contre la boucle directe
  puis cherche la plus grande taille de lot qui tient en VRAM.
  Ne modifie rien.

  Doit se trouver dans le MÊME dossier que acceleration_qwen.py.

  LANCEMENT (depuis le dossier de l'application, studio fermé) :
      qwen_env\\Scripts\\python.exe diagnostic_vitesse.py

  Durée : 4 à 8 minutes. Copie le bloc « RÉSUMÉ » final et renvoie-le.
================================================================================
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

PHRASES = [
    "Problème :",
    "La réponse tient en un mot :",
    "Il faut prendre la mesure de la situation ainsi créée.",
    "Le train quitte la gare à sept heures précises chaque matin.",
    "Une carte, un carnet, une lampe de poche : tout le nécessaire tenait dans un sac.",
    "Le jardin produisait à peine de quoi nourrir la famille, ce qui obligeait chacun à travailler dès l'aube.",
    "Le vieux phare veillait sur la baie depuis plus d'un siècle, et les marins le saluaient en passant au large.",
    "Plus de deux cents ouvriers, trois ponts métalliques et une gare entière furent nécessaires pour relier enfin les deux vallées.",
]


def main():
    ici = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, ici)
    try:
        from acceleration_qwen import installer_acceleration, retirer_acceleration
    except ImportError:
        print("✗ acceleration_qwen.py doit être dans le même dossier que ce script.")
        sys.exit(1)

    conf = {}
    chemin_conf = os.path.join(ici, "qwen_tts_studio_config.json")
    if os.path.isfile(chemin_conf):
        import json
        with open(chemin_conf, encoding="utf-8") as f:
            conf = json.load(f)
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default=conf.get("ref_audio") or "",
                   help="voix de référence (défaut : celle du studio)")
    p.add_argument("--ref-text", default=None,
                   help="transcription de la voix (défaut : celle du studio)")
    p.add_argument("--modele", default=conf.get("model_id")
                   or "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--langue", default=conf.get("language") or "French")
    p.add_argument("--lots", default="8,12,16,20",
                   help="tailles de lot à essayer (séparées par des virgules)")
    args = p.parse_args()
    if args.ref_text is None:
        # La transcription du studio ne vaut que pour la voix du studio.
        meme_voix = args.ref == (conf.get("ref_audio") or "")
        args.ref_text = conf.get("ref_text", "") if meme_voix else ""

    import torch
    import numpy as np
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    if not torch.cuda.is_available():
        print("✗ CUDA indisponible : ce diagnostic n'a de sens que sur GPU.")
        sys.exit(1)
    if not args.ref or not os.path.isfile(args.ref):
        print(f"✗ Voix de référence introuvable : {args.ref or '(aucune)'}\n"
              "  Lance d'abord un rendu dans le studio, ou précise --ref fichier.wav")
        sys.exit(1)

    resume = []

    def dire(ligne=""):
        print(ligne)
        resume.append(ligne)

    prop = torch.cuda.get_device_properties(0)
    total_go = prop.total_memory / 1e9
    dire(f"GPU        : {torch.cuda.get_device_name(0)}  ({total_go:.1f} Go)")

    print("\nChargement du modèle…")
    modele = Qwen3TTSModel.from_pretrained(args.modele, dtype=torch.float16, device_map="cuda:0")
    info = sf.info(args.ref)
    dire(f"Référence  : {os.path.basename(args.ref)} — {info.duration:.1f} s, "
         f"transcription {len(args.ref_text)} car.")
    prompt = modele.create_voice_clone_prompt(
        ref_audio=args.ref, ref_text=args.ref_text or None,
        x_vector_only_mode=not args.ref_text)

    def generer(textes, graine=1234):
        torch.manual_seed(graine)
        torch.cuda.synchronize()
        t = time.perf_counter()
        wavs, sr = modele.generate_voice_clone(
            text=textes, language=args.langue, voice_clone_prompt=prompt)
        torch.cuda.synchronize()
        return time.perf_counter() - t, sum(len(w) for w in wavs) / sr, wavs

    def identiques(a, b):
        return len(a) == len(b) and all(
            len(x) == len(y) and np.array_equal(x, y) for x, y in zip(a, b))

    print("Préchauffage…")
    generer(PHRASES[:2])

    # --- 1. Les trois niveaux, lot de 8 ---------------------------------------
    print("\n[1/2] Origine / boucle directe / CUDA Graph (lot de 8)…")
    t_o, a_o, w_o = generer(PHRASES)
    dire(f"Origine    : {t_o:5.1f} s  ({a_o / t_o:.2f} s d'audio par seconde)")

    installer_acceleration(modele, torch, graphes=False)
    t_d, a_d, w_d = generer(PHRASES)
    dire(f"Directe    : {t_d:5.1f} s  ({a_d / t_d:.2f} s/s, x{t_o / t_d:.2f})   "
         f"audio identique à l'origine : {'OUI' if identiques(w_o, w_d) else 'NON'}")
    retirer_acceleration(modele)

    t0 = time.perf_counter()
    libelle = installer_acceleration(modele, torch, graphes=True)
    preparation = time.perf_counter() - t0
    dire(f"Graphe     : {libelle}  (préparation {preparation:.1f} s)")
    generer(PHRASES[:3])                       # enregistre les graphes utiles
    generer(PHRASES)
    t_g, a_g, w_g = generer(PHRASES)
    dire(f"Graphe     : {t_g:5.1f} s  ({a_g / t_g:.2f} s/s, x{t_o / t_g:.2f})   "
         f"audio identique : {'OUI' if identiques(w_o, w_g) else 'non (tirage aléatoire, normal)'}")

    # --- 2. Taille de lot (avec la meilleure accélération) --------------------
    print("\n[2/2] Taille de lot…")
    textes = PHRASES * 4
    for taille in [int(x) for x in args.lots.split(",") if x.strip()]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            generer(textes[:taille])           # 1er passage : enregistre le graphe
            t, a, _ = generer(textes[:taille])
        except Exception as e:  # noqa: BLE001
            dire(f"Lot {taille:>2}     : échec ({str(e).splitlines()[0][:60]})")
            torch.cuda.empty_cache()
            continue
        pic = torch.cuda.max_memory_reserved() / 1e9
        deborde = "  ⚠ DÉBORDE sur la mémoire système" if pic > 0.95 * total_go else ""
        dire(f"Lot {taille:>2}     : {a / t:.2f} s d'audio par seconde   "
             f"(VRAM pic {pic:.1f} Go){deborde}")

    print("\n" + "=" * 70)
    print("  RÉSUMÉ — copie tout ce bloc et renvoie-le")
    print("=" * 70)
    for ligne in resume:
        print(ligne)
    print("=" * 70)


if __name__ == "__main__":
    main()
