# Comment le studio va 20 fois plus vite

Cette note explique les optimisations du studio par rapport à l'utilisation directe de Qwen3-TTS. Elle s'adresse aux personnes curieuses ou qui souhaitent contribuer. Pour se servir de l'application, il n'est pas nécessaire de la lire.

Toutes les mesures ont été faites sur une NVIDIA RTX A4500 (20 Go), sous Windows, avec une voix de référence de 29 s accompagnée de sa transcription.

## Le constat de départ

Qwen3-TTS produit le son par **trames** de 1/12,5 seconde. Pour chaque trame :

1. le **modèle principal** (*talker*) choisit un premier code ;
2. un **sous-modèle** (*code predictor*) en déduit 15 autres, un par un.

Une phrase de 10 secondes représente donc 125 trames, soit environ 2 000 passages dans le sous-modèle. Chaque passage lance des dizaines de petites opérations sur la carte graphique. Une mesure détaillée a montré que **le sous-modèle prenait 67 % du temps**, alors qu'il est minuscule : la carte graphique passait l'essentiel de son temps à **attendre le processeur**, qui lui envoyait les opérations une par une.

## Les six optimisations

### 1. Génération par lots (`qwen_tts_studio.py`)

Puisque la carte graphique attend, autant lui confier plusieurs phrases à la fois : 8, 16 ou 32 phrases coûtent presque le même temps qu'une seule. Dans chaque fenêtre de phrases consécutives, celles de longueur proche sont regroupées, car un lot dure aussi longtemps que sa phrase la plus longue. Le rendu final est ensuite remis dans l'ordre du texte.

### 2. Analyse unique de la voix de référence

La version d'origine relisait, rééchantillonnait et ré-encodait le fichier de référence **à chaque phrase**. Il est maintenant analysé une seule fois par rendu (`create_voice_clone_prompt`).

### 3. Boucle directe du sous-modèle (`acceleration_qwen.py`)

L'appel `generate()` de la bibliothèque transformers, utilisé pour produire les 15 codes de chaque trame, est remplacé par une boucle minimale qui fait **exactement le même calcul** : même cache, mêmes têtes, même tirage (température, puis top-k, puis top-p). Le tirage emploie la loi exponentielle, que `torch.multinomial` utilise lui-même pour un seul échantillon. Résultat vérifié : **audio identique au bit près**, gain ×1,2.

### 4. CUDA Graph du sous-modèle (`acceleration_qwen.py`)

Un CUDA Graph enregistre une séquence d'opérations GPU, puis la **rejoue en un seul lancement**. Les 15 codes d'une trame sont enregistrés une fois par taille de lot. Pour que ce soit possible :
- le cache d'attention est **préalloué** à sa taille maximale de 16 positions (`_CacheFixe`) ;
- les positions et les tables de rotation (RoPE) sont calculées à l'avance ;
- le tirage n'utilise aucune opération qui oblige la carte à attendre le processeur.

**Contrôle à chaque chargement** :
1. le graphe doit rejouer **exactement** le même calcul exécuté normalement, en mode déterministe ;
2. ce calcul doit concorder à au moins 90 % avec la bibliothèque d'origine ;
3. la mémoire libre est occupée par des valeurs témoins avant de rejouer le graphe, puis on vérifie qu'il n'a écrit que dans ses propres tampons.

Si l'une de ces vérifications échoue, le studio repasse à la boucle directe (optimisation 3) et l'indique dans le journal. Les 20 premiers rejeux de chaque graphe sont aussi vérifiés : un code hors vocabulaire provoque le même retour à la boucle directe.

Gain mesuré : **×2,9 par rapport au code d'origine**, avec un lot de 8.

### 5. Décodage audio phrase par phrase

Pour garder une voix cohérente, la bibliothèque **redécode la voix de référence avec chaque phrase**, puis retire cette partie du résultat. Avec un lot de 8, ce décodage occupait **5,9 Go** de mémoire vidéo, contre 1,1 Go pour toute la génération des codes. En décodant les phrases une par une, ce pic tombe à **0,74 Go**, pour le même temps de calcul. L'écart avec le décodage groupé est au plus de 0,008 (environ −42 dB), localisé et inaudible : il vient du remplissage qu'ajoute le décodage groupé pour égaliser les longueurs.

C'est ce qui permet de passer de lots de 8 à des lots de 32.

| Lot | Audio par seconde de calcul | Mémoire vidéo |
|---|---|---|
| 8 | 3,4 s | 6,1 Go |
| 16 | 5,3 s | 7,2 Go |
| 32 | 7,4 s | 11,8 Go |

### 6. Garde-fous

- **Durée maximale par phrase** : il arrive que le modèle ne s'arrête pas et génère jusqu'à sa limite d'origine de 2 048 trames (2 min 44 s). Dans ce cas, tout le lot attend. La durée est donc plafonnée à environ 4 fois la durée normale du texte : 2,5 trames par caractère plus 40, alors qu'on observe environ 0,6 trame par caractère en français. Une phrase qui atteint ce plafond est signalée dans le journal.
- **Réduction automatique du lot** : si la mémoire vidéo réservée dépasse 95 % de la mémoire de la carte, le lot est divisé par deux pour la suite du rendu. Sous Windows, la carte déborderait sinon sur la mémoire système, ce qui ralentit le rendu jusqu'à 10 fois sans aucun message d'erreur.
- **Reprise en cas d'erreur** : si un lot échoue, ses phrases sont reprises une par une.

## Bilan

| Étape | Audio par seconde de calcul |
|---|---|
| Code d'origine, une phrase à la fois | 0,37 s |
| + génération par lots de 8 | 1,25 s |
| + CUDA Graph | 3,4 s |
| + décodage phrase par phrase et lots de 32 | **7,4 s** |

## Désactiver, mesurer

- `ACCELERATION = False`, en tête de `qwen_tts_studio.py`, restaure le comportement de la bibliothèque d'origine. La génération par lots est conservée.
- `diagnostic_vitesse.py` compare l'origine, la boucle directe et le CUDA Graph, puis teste plusieurs tailles de lot.
- `diagnostic_vram.py` détaille la mémoire utilisée par chaque étape.

## Compatibilité

`acceleration_qwen.py` s'appuie sur la structure interne de **qwen-tts 0.1.1** et de **transformers 4.57.3**, les versions figées dans `requirements.txt`. Avec d'autres versions, le contrôle au chargement écarte le CUDA Graph s'il ne se comporte plus correctement. Si la structure du modèle a changé, l'accélération est désactivée et le journal l'indique.
