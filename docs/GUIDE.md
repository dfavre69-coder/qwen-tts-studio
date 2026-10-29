# Guide d'utilisation — Qwen3-TTS Voice Studio

Ce guide s'adresse à tout le monde, y compris aux personnes qui n'ont jamais utilisé Python. Les termes techniques sont expliqués la première fois qu'ils apparaissent.

## Sommaire

1. [Les mots à connaître](#1-les-mots-à-connaître)
2. [Installation pas à pas](#2-installation-pas-à-pas)
3. [Préparer une bonne voix de référence](#3-préparer-une-bonne-voix-de-référence)
4. [L'application, onglet par onglet](#4-lapplication-onglet-par-onglet)
5. [Pendant le rendu : lire le journal](#5-pendant-le-rendu--lire-le-journal)
6. [Régler la vitesse selon votre carte graphique](#6-régler-la-vitesse-selon-votre-carte-graphique)
7. [Bien préparer son texte](#7-bien-préparer-son-texte)
8. [Problèmes courants](#8-problèmes-courants)
9. [Questions fréquentes](#9-questions-fréquentes)

---

## 1. Les mots à connaître

| Terme | Signification |
|---|---|
| **Voix de référence** | Court enregistrement (10 à 30 s) de la voix à imiter. |
| **Transcription** | Le texte prononcé dans la voix de référence, mot pour mot. |
| **Modèle** | Le « cerveau » qui fabrique la voix : Qwen3-TTS 1.7B, environ 4,5 Go, téléchargé une seule fois. |
| **Chunk (morceau)** | Le texte est découpé en morceaux d'une phrase environ, générés séparément puis assemblés. |
| **Lot** | Groupe de morceaux générés en même temps par la carte graphique. Plus le lot est grand, plus c'est rapide, tant que la mémoire suffit. |
| **VRAM** | La mémoire de la carte graphique. C'est elle qui limite la taille des lots. |
| **GPU** | La carte graphique. L'application a besoin d'une carte NVIDIA pour être rapide. |

---

## 2. Installation pas à pas

### 2.1 Télécharger l'application

1. Sur la page GitHub du projet, cliquez sur le bouton vert **Code**, puis **Download ZIP**.
2. Décompressez le fichier dans un dossier **simple et local**, par exemple `D:\qwen-tts-studio`.
   - Évitez le Bureau, OneDrive, Google Drive ou Dropbox : la synchronisation de plusieurs gigaoctets de fichiers ralentit tout et peut bloquer l'installation.
   - Prévoyez **15 Go** d'espace libre.

### 2.2 Lancer l'installation

Double-cliquez sur **`installer.bat`**. Une fenêtre noire s'ouvre et affiche cinq étapes :

| Étape | Ce qui se passe | Durée |
|---|---|---|
| **1/5 Python 3.12** | Cherche Python 3.12. S'il est absent, propose de l'installer : tapez `O` pour oui. | 1 à 3 min |
| **2/5 Environnement** | Crée le dossier `qwen_env`, qui contient tous les composants. Il ne touche à rien d'autre sur votre ordinateur. | quelques secondes |
| **3/5 Paquets** | Télécharge PyTorch (le moteur de calcul, environ 2,5 Go) puis les autres composants. | 5 à 20 min |
| **4/5 ffmpeg** | Vérifie ffmpeg, l'outil qui fabrique les MP3 et les vidéos. Propose de l'installer s'il manque. | 1 à 2 min |
| **5/5 Modèle** | Propose de télécharger le modèle maintenant (environ 4,5 Go). Si vous répondez `N`, il sera téléchargé au premier rendu. | 2 à 10 min |

À la fin, une ligne doit afficher le nom de votre carte graphique, par exemple :

```
PyTorch 2.5.1+cu121 | GPU : NVIDIA GeForce RTX 4070
```

Si elle affiche **AUCUN**, consultez la section [Problèmes courants](#8-problèmes-courants).

> 💡 `installer.bat` peut être relancé sans risque : il complète ce qui manque et ne détruit rien.

### 2.3 Lancer l'application

Double-cliquez sur **`lancer_studio.bat`**. La fenêtre de l'application s'ouvre immédiatement. Le modèle n'est chargé qu'au premier rendu, ce qui prend environ une minute, puis il reste en mémoire tant que l'application est ouverte.

### 2.4 Désinstaller

Supprimez simplement le dossier de l'application. Le modèle téléchargé se trouve dans `C:\Users\<votre nom>\.cache\huggingface` : supprimez aussi ce dossier pour récupérer les 4,5 Go.

---

## 3. Préparer une bonne voix de référence

La qualité de la voix clonée dépend presque entièrement de cet enregistrement.

**À faire**
- ✅ **10 à 30 secondes** de parole continue. Une référence plus courte prend aussi moins de mémoire et permet des lots plus grands.
- ✅ **Une seule personne**, au calme, sans musique ni écho.
- ✅ Le **ton que vous voulez obtenir** : posé pour un livre audio, vivant pour une vidéo. Le modèle reprend le ton de l'extrait.
- ✅ Un fichier **WAV** si possible. MP3 et FLAC sont aussi acceptés.
- ✅ Une **transcription exacte**, mot pour mot, avec la ponctuation, collée dans le champ prévu.

**À éviter**
- ❌ Un extrait coupé au milieu d'un mot.
- ❌ Une transcription approximative : le modèle s'en sert pour comprendre comment la voix prononce chaque son.
- ❌ La voix de quelqu'un sans son accord.

Si vous laissez la transcription vide, l'application utilise seulement l'**empreinte** de la voix. Ça fonctionne, mais le résultat est nettement moins fidèle.

---

## 4. L'application, onglet par onglet

En bas de la fenêtre, quel que soit l'onglet :
- **Aperçu du découpage** : montre comment le texte sera découpé et nettoyé, sans lancer le modèle. Utile pour repérer un problème avant un long rendu.
- **▶ Générer l'audio** : lance le rendu.
- **⏹ Arrêter** : interrompt le rendu. Ce qui a déjà été produit est enregistré avec le suffixe `-partiel`.
- **📂 Dossier** et **🔊 Lire** : ouvrent le dossier de sortie ou le dernier fichier produit.

### Onglet « Entrées »

| Champ | Explication |
|---|---|
| **Voix de référence** | Le fichier audio de la voix à imiter. |
| **Transcription exacte** | Le texte prononcé dans cet extrait. |
| **Fichier texte à lire** | Votre texte : `.docx` (Word), `.md` ou `.txt`. |
| **Dossier de sortie** | Où enregistrer le résultat. Par défaut, le dossier du texte. |
| **Nom de base du rendu** | Le nom du fichier produit. Cochez **Horodater** pour ne jamais écraser un rendu précédent : la date et l'heure sont alors ajoutées au nom. |

### Onglet « Voix & Découpage »

| Réglage | Conseil |
|---|---|
| **Ton / émotion** | Sans effet avec les modèles *Base* utilisés pour le clonage : le ton vient de la voix de référence. |
| **Langue de synthèse** | La langue du texte à lire. |
| **Taille max d'un chunk** | 350 caractères est un bon compromis. Avec des phrases plus courtes, les intonations sont plus naturelles, mais le rendu est un peu plus lent. |
| **Phrases générées en parallèle (lot)** | Voir la [section 6](#6-régler-la-vitesse-selon-votre-carte-graphique). |
| **Silences** | Pauses ajoutées après une virgule, un point ou un paragraphe, en secondes. |
| **Niveau cible (dBFS)** | Volume maximal de chaque phrase. −3 est fort sans saturer. |
| **Fréquence supplémentaire** | Crée aussi une version en 44 100 Hz (standard des CD et des podcasts) ou en 48 000 Hz (standard de la vidéo). |

### Onglet « Export MP3 »

Cochez **Générer aussi un fichier MP3** pour obtenir un fichier plus léger. Le **mono** à 128 ou 192 kbit/s convient très bien à la voix. Si ffmpeg n'est pas détecté, indiquez son emplacement ou cliquez sur **🔍 Détecter ffmpeg**.

### Onglet « Vidéo »

Cet onglet fabrique une vidéo à partir d'**une image fixe** (une couverture, par exemple) et d'**une piste audio**.
- **↩ Utiliser le dernier rendu TTS** reprend automatiquement le fichier audio que vous venez de générer.
- **Enchaîner automatiquement** lance la vidéo dès que l'audio est terminé.
- **Résolution** : 1920 × 1080 pour YouTube, 1080 × 1920 pour les Shorts et les Reels.
- **Images par seconde** : pour une image fixe, 1 ou 2 suffisent, et la vidéo est beaucoup plus rapide à produire.

### Onglet « Fusion & Chapitres »

Cet onglet sert à **assembler plusieurs fichiers audio**, par exemple les chapitres d'un livre lus séparément.

1. **➕ Ajouter…** les fichiers, puis ordonnez-les avec **▲** et **▼**.
2. Donnez un **libellé** à chaque segment : double-cliquez dessus pour le modifier. Ce libellé devient le titre du chapitre.
3. Réglez le **silence entre segments** et la **normalisation** : « Par segment » harmonise le volume des fichiers entre eux. La cible **−16 LUFS** convient aux podcasts, **−14 LUFS** à YouTube et Spotify.
4. Cliquez sur **🔗 Fusionner**.

Le **chapitrage** est généré automatiquement : copiez-le dans la description YouTube pour obtenir des chapitres cliquables. L'application vous avertit si les règles de YouTube ne sont pas respectées : le premier chapitre doit commencer à 00:00, il faut au moins 3 chapitres, et chacun doit durer au moins 10 s. Les chapitres peuvent aussi être intégrés au fichier lui-même : ils sont alors reconnus par les lecteurs de podcasts et de livres audio.

### Onglet « Système »

| Réglage | Explication |
|---|---|
| **Identifiant du modèle** | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` par défaut. La version `0.6B-Base` est plus légère mais moins fidèle. |
| **Device** | `auto` choisit la carte graphique si elle est disponible. |
| **Token Hugging Face** | Inutile pour les modèles Qwen3-TTS publics. |
| **🔍 Vérifier l'environnement** | Affiche les versions installées et la carte graphique détectée. |
| **💾 Sauver / 📁 Charger config** | Enregistre tous vos réglages dans un fichier, pour passer d'un projet à l'autre. |

Vos réglages sont aussi enregistrés automatiquement à la fermeture, dans `qwen_tts_studio_config.json`.

---

## 5. Pendant le rendu : lire le journal

Le panneau **Journal** affiche en direct ce que fait l'application.

| Message | Signification |
|---|---|
| `Accélération du sous-modèle : CUDA Graph (contrôlé, 100 % de concordance)` | L'accélération maximale est active. |
| `Accélération du sous-modèle : boucle directe (…)` | L'accélération maximale n'a pas pu être activée sur votre carte. Le rendu fonctionne quand même, mais plus lentement. La raison est indiquée entre parenthèses. |
| `[  54/346] (117 car.) ¶ : …` | Le morceau 54 sur 346 est en cours. Le symbole ¶ signale une fin de paragraphe. Les numéros n'apparaissent pas dans l'ordre, c'est normal : les morceaux de longueur proche sont regroupés dans un même lot. |
| `⏱ 32 phrases en 9.8 s` | La vitesse de chaque série de lots. |
| `💾 VRAM utilisée : 11.8 Go` | La mémoire de la carte graphique occupée. |
| `⚠ VRAM pleine … lot réduit à 16` | La mémoire était saturée : l'application a réduit le lot d'elle-même. En fin de rendu, elle vous indique la valeur à retenir. |
| `⚠ Phrase coupée à la durée maximale` | Le modèle a « bouclé » sur cette phrase et ne s'arrêtait plus. Elle a été coupée : écoutez ce passage et relancez le rendu si besoin. |

Le rendu terminé, l'application indique la durée de l'audio et le temps de calcul.

---

## 6. Régler la vitesse selon votre carte graphique

La vitesse dépend surtout de la **taille des lots**. Elle se règle tout en haut du fichier `qwen_tts_studio.py`, que vous pouvez ouvrir avec le Bloc-notes :

```python
TAILLE_LOT = 32
```

Valeurs indicatives, avec une voix de référence d'environ 30 s :

| Mémoire de la carte | `TAILLE_LOT` conseillé |
|---|---|
| 8 Go | 8 |
| 12 Go | 16 |
| 16 Go et plus | 32 |

Pas d'inquiétude si vous visez trop haut : en cas de saturation, l'application réduit le lot d'elle-même pendant le rendu, puis vous indique la valeur à retenir.

**Pour mesurer précisément**, fermez l'application et lancez dans une invite de commandes ouverte dans le dossier de l'application :

```
qwen_env\Scripts\python.exe diagnostic_vitesse.py --lots 8,16,24,32
```

Au bout de quelques minutes, un résumé indique la vitesse et la mémoire utilisée pour chaque taille de lot. Si une ligne affiche **DÉBORDE**, cette taille est trop grande pour votre carte.

**Pour désactiver les optimisations**, par exemple pour comparer ou pour diagnostiquer un problème, mettez :

```python
ACCELERATION = False
```

---

## 7. Bien préparer son texte

- **Les intertitres** doivent se terminer par un point. Sinon, ils sont lus d'un trait avec la phrase suivante.
- **Les nombres** sont écrits automatiquement en toutes lettres, jusqu'à 999 999. Au-delà, mieux vaut les écrire vous-même.
- **Les abréviations** (M., Dr, etc.) sont lues telles quelles : écrivez-les en entier.
- **Les sigles** sont lus comme des mots s'ils sont prononçables. Sinon, écrivez-les avec des points (`S.N.C.F.`) ou en toutes lettres.
- Le **Markdown** (titres, gras, listes, liens) et les **emojis** sont retirés automatiquement.
- Utilisez **Aperçu du découpage** pour vérifier le résultat avant de lancer un long rendu.

---

## 8. Problèmes courants

| Symptôme | Cause probable | Solution |
|---|---|---|
| Messages `SoX could not be found`, `flash-attn is not installed` ou `Setting pad_token_id` | Avertissements de la bibliothèque Qwen3-TTS. | **À ignorer**, ils n'ont aucun effet. |
| `GPU : AUCUN` à l'installation, ou rendu extrêmement lent | Pas de carte NVIDIA, ou pilote trop ancien. | Mettez à jour le pilote NVIDIA (GeForce Experience ou nvidia.com), puis relancez `installer.bat`. |
| Le premier rendu reste longtemps sur « Chargement du modèle » | Le modèle (4,5 Go) est en cours de téléchargement. | Patientez. Ça n'arrive qu'une seule fois. |
| `⚠ VRAM pleine` à chaque rendu | `TAILLE_LOT` trop grand pour votre carte. | Prenez la valeur que l'application vous indique en fin de rendu. |
| Le rendu ralentit fortement en cours de route | Une autre application utilise la carte graphique (jeu, montage vidéo, autre IA). | Fermez-la pendant le rendu. |
| Pas de MP3 ni de vidéo | ffmpeg est introuvable. | Onglet **Export MP3** → **🔍 Détecter ffmpeg**, ou relancez `installer.bat`. |
| La voix ne ressemble pas assez | Référence bruitée, trop courte, ou transcription inexacte. | Voir la [section 3](#3-préparer-une-bonne-voix-de-référence). |
| Une phrase est répétée, étirée ou coupée | Le modèle a « bouclé » : ça arrive rarement, c'est dû au tirage aléatoire. | Relancez le rendu. Si le problème revient, reformulez ou découpez la phrase. |
| `lancer_studio.bat` affiche « L'environnement n'est pas installé » | `installer.bat` n'a pas été lancé, ou n'a pas abouti. | Lancez ou relancez `installer.bat`. |

---

## 9. Questions fréquentes

**Faut-il Internet pour s'en servir ?**
Seulement pour l'installation et le premier téléchargement du modèle. Ensuite, tout fonctionne hors ligne, et rien n'est envoyé sur Internet.

**Quelles langues sont prises en charge ?**
Français, anglais, chinois, espagnol, allemand, italien, japonais, coréen, portugais et russe. Le nettoyage automatique des nombres est prévu pour le français.

**Est-ce que ça fonctionne sur Mac ou Linux ?**
L'application est conçue et testée pour Windows. Le code Python peut fonctionner sous Linux avec une carte NVIDIA, mais les scripts d'installation `.bat` sont propres à Windows.

**Puis-je vendre les livres audio produits ?**
La licence Apache 2.0 de l'application et du modèle Qwen3-TTS le permet. En revanche, vous devez avoir le **droit d'utiliser la voix** et le **droit d'exploiter le texte** : une œuvre du domaine public, la vôtre, ou une œuvre pour laquelle vous avez une autorisation.

**Pourquoi deux rendus du même texte ne sont-ils pas identiques ?**
Le modèle introduit volontairement un peu de hasard pour que la voix sonne naturelle. Chaque rendu est donc légèrement différent.
