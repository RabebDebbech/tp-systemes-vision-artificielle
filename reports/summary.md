# Synthèse du projet

## Dataset
- 2527 images
- 6 classes : cardboard, glass, metal, paper, plastic, trash

## Préparation
- inventaire des fichiers
- création des splits train / val / test stratifiés
- manifest enrichi avec métadonnées
- redimensionnement des images en 224x224

## Modèles
### Baseline
- HOG + Linear SVM

### CNN
- ResNet18 pretrained
- config baseline
- config improved

## Déploiement
- API FastAPI avec endpoint `/predict`
- conteneur Docker

## Fichiers importants
- `data/processed/splits.csv`
- `data/processed/manifest.parquet`
- `data/processed/manifest_224.parquet`
- `models/hog_svm.joblib`
- `models/cnn_baseline/best_model.pt`
- `models/cnn_improved/best_model.pt`
- `src/api/main.py`