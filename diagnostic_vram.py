# -*- coding: utf-8 -*-
# Copyright 2026 Denis Favre
# SPDX-License-Identifier: Apache-2.0
"""
================================================================================
  DIAGNOSTIC VRAM — Qwen3-TTS
================================================================================
  Mesure où part la mémoire GPU pendant un rendu :
      • modèle chargé (poids)
      • génération des codes (modèle principal + sous-modèle)
      • décodage audio (codes -> forme d'onde)
  avec des lots de 4 et 8, puis essaie un décodage par petits groupes.
  Ne modifie rien.

  Doit se trouver dans le MÊME dossier que acceleration_qwen.py.

  LANCEMENT (depuis le dossier de l'application, studio fermé) :
      qwen_env\\Scripts\\python.exe diagnostic_vram.py [voix.wav]

  Durée : 2 à 4 minutes. Copie le bloc « RÉSUMÉ » final et renvoie-le.
================================================================================
"""

import os
import sys
import time
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

GO = 1e9


def main():
    ici = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, ici)

    import json
    conf = {}
    chemin_conf = os.path.join(ici, "qwen_tts_studio_config.json")
    if os.path.isfile(chemin_conf):
        with open(chemin_conf, encoding="utf-8") as f:
            conf = json.load(f)
    # Voix : 1er argument, sinon celle du studio
    ref = sys.argv[1] if len(sys.argv) > 1 else (conf.get("ref_audio") or "")
    # La transcription du studio ne vaut que pour la voix du studio.
    ref_text = conf.get("ref_text", "") if len(sys.argv) <= 1 else ""
    modele_id = conf.get("model_id") or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    langue = conf.get("language") or "French"

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    if not torch.cuda.is_available():
        print("✗ CUDA indisponible.")
        sys.exit(1)
    if not ref or not os.path.isfile(ref):
        print(f"✗ Voix de référence introuvable : {ref or '(aucune)'}\n"
              "  Lance d'abord un rendu dans le studio, ou : diagnostic_vram.py voix.wav")
        sys.exit(1)

    resume = []

    def dire(ligne=""):
        print(ligne)
        resume.append(ligne)

    total = torch.cuda.get_device_properties(0).total_memory / GO
    dire(f"GPU        : {torch.cuda.get_device_name(0)}  ({total:.1f} Go)")
    dire(f"Référence  : {os.path.basename(ref)} — {sf.info(ref).duration:.1f} s, "
         f"transcription {len(ref_text)} car.")

    print("\nChargement du modèle…")
    modele = Qwen3TTSModel.from_pretrained(modele_id, dtype=torch.float16, device_map="cuda:0")
    try:
        from acceleration_qwen import installer_acceleration
        dire(f"Accélération : {installer_acceleration(modele, torch, graphes=True)}")
    except ImportError:
        dire("Accélération : absente (acceleration_qwen.py introuvable)")

    prompt = modele.create_voice_clone_prompt(
        ref_audio=ref, ref_text=ref_text or None, x_vector_only_mode=not ref_text)
    torch.cuda.synchronize()
    dire(f"Modèle seul        : {torch.cuda.memory_allocated() / GO:5.1f} Go")
    n_ref = 0
    try:
        n_ref = int(prompt[0].ref_code.shape[0])
        dire(f"Codes de référence : {n_ref} trames ({n_ref / 12.5:.1f} s), "
             "redécodées avec CHAQUE phrase")
    except Exception:  # noqa: BLE001
        pass

    # --- Instrumentation -----------------------------------------------------
    mesures = {}
    gen_orig = modele.model.generate
    tok = modele.model.speech_tokenizer
    dec_orig = tok.decode

    def gen_mesure(*a, **k):
        torch.cuda.synchronize()
        avant = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        r = gen_orig(*a, **k)
        torch.cuda.synchronize()
        mesures["gen_pic"] = (torch.cuda.max_memory_allocated() - avant) / GO
        mesures["gen_reste"] = (torch.cuda.memory_allocated() - avant) / GO
        return r

    def dec_mesure(encoded, *a, **k):
        mesures["codes"] = encoded
        torch.cuda.synchronize()
        avant = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        r = dec_orig(encoded, *a, **k)
        torch.cuda.synchronize()
        mesures["dec_temps"] = time.perf_counter() - t
        mesures["dec_pic"] = (torch.cuda.max_memory_allocated() - avant) / GO
        return r

    modele.model.generate = gen_mesure
    tok.decode = dec_mesure

    def rendu(textes):
        torch.manual_seed(1234)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        modele.generate_voice_clone(text=textes, language=langue, voice_clone_prompt=prompt)
        torch.cuda.synchronize()
        return time.perf_counter() - t, torch.cuda.max_memory_reserved() / GO

    print("Préchauffage…")
    rendu(PHRASES[:2])

    # --- 1. Lots de 4 et 8 ----------------------------------------------------
    print("\n[1/2] Mémoire par étape…")
    for lot in (4, 8):
        duree, reserve = rendu(PHRASES[:lot])
        dire(f"Lot {lot} : génération pic +{mesures['gen_pic']:.2f} Go "
             f"(dont {mesures['gen_reste']:.2f} Go conservés après) | "
             f"décodage pic +{mesures['dec_pic']:.2f} Go en {mesures['dec_temps']:.1f} s | "
             f"total réservé {reserve:.1f} Go | {duree:.1f} s")

    # --- 2. Décodage par petits groupes (même codes que le lot de 8) ---------
    print("\n[2/2] Décodage par petits groupes…")
    codes = mesures["codes"]
    tok.decode = dec_orig
    ref_wavs, _ = dec_orig(codes)
    for taille in (1, 2, 4):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        avant = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        wavs = []
        for i in range(0, len(codes), taille):
            w, _ = dec_orig(codes[i:i + taille])
            wavs.extend(w)
        torch.cuda.synchronize()
        duree = time.perf_counter() - t
        pic = (torch.cuda.max_memory_allocated() - avant) / GO
        import numpy as np
        ecart = max(float(np.max(np.abs(a[:min(len(a), len(b))] - b[:min(len(a), len(b))])))
                    for a, b in zip(ref_wavs, wavs))
        memes_longueurs = all(len(a) == len(b) for a, b in zip(ref_wavs, wavs))
        dire(f"Décodage par {taille} : pic +{pic:.2f} Go en {duree:.1f} s | "
             f"écart max avec le décodage groupé {ecart:.1e}"
             f"{'' if memes_longueurs else ' (longueurs différentes)'}")

    print("\n" + "=" * 70)
    print("  RÉSUMÉ — copie tout ce bloc et renvoie-le")
    print("=" * 70)
    for ligne in resume:
        print(ligne)
    print("=" * 70)


if __name__ == "__main__":
    main()
