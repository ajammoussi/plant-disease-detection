# 🌿 Plant Disease Detection — Data Preprocessing

## Structure du projet (partie prétraitement)

```
plant_disease_detection/
│
├── data/                          ← données (générée à l'exécution)
│   ├── plantdisease.zip           ← archive téléchargée
│   └── raw/                       ← images extraites (classes/)
│
├── notebooks/
│   └── 01_data_preprocessing.ipynb   ← notebook principal
│
├── outputs/
│   └── figures/                   ← toutes les illustrations sauvegardées
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             ← téléchargement, scan, chargement
│   ├── eda.py                     ← analyse exploratoire & visualisations
│   ├── data_cleaning.py           ← nettoyage (corrupt, doublons, qualité)
│   └── preprocessing.py           ← resize, CLAHE, denoise, normalisation
│
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancement du notebook

```bash
cd notebooks
jupyter lab 01_data_preprocessing.ipynb
```

## Pipeline de prétraitement

| Étape | Module | Fonction clé |
|-------|--------|-------------|
| Téléchargement | `data_loader` | `download_dataset()` |
| Scan classes | `data_loader` | `scan_dataset()` |
| EDA | `eda` | `plot_class_distribution()`, `plot_sample_grid()`, … |
| Nettoyage | `data_cleaning` | `find_corrupt_images()`, `find_exact_duplicates()` |
| Redimensionnement | `preprocessing` | `resize_image()` |
| Amélioration | `preprocessing` | `apply_clahe()`, `apply_bilateral_denoise()` |
| Pipeline complet | `preprocessing` | `full_enhancement_pipeline()` |
| Normalisation | `preprocessing` | `normalize_imagenet()` |
| Avant/Après | `preprocessing` | `plot_before_after()`, `plot_pipeline_stages()` |

## Dataset

- **Source** : [PlantVillage — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **URL API** : `https://www.kaggle.com/api/v1/datasets/download/emmarex/plantdisease`
- ~54 000 images, 38 classes (espèces × maladies)
