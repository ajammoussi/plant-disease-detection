import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage dataset" / "color"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"  # Nouveau dossier pour les matrices

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ==========================================
# 2. PRÉPARATION DES DONNÉES (Validation Uniquement)
# ==========================================
def get_val_dataloader_and_classes(batch_size=32, val_split=0.2):
    print("Chargement des données de validation...")

    # Uniquement la transformation de validation
    val_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(str(PROCESSED_DIR), transform=val_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Utilisation du MÊME seed (42) pour garantir les mêmes images de validation
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return val_loader, class_names, num_classes


# ==========================================
# 3. INITIALISATION DU MODÈLE
# ==========================================
def initialize_model0(model_name, num_classes):
    if model_name == "resnet50":
        # Pas besoin des poids pré-entraînés ici puisqu'on charge nos propres poids
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2()
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Modèle non supporté")

    return model.to(device)

def initialize_model(model_name, num_classes):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Modèle non supporté")
    return model.to(device)



# ==========================================
# 4. ÉVALUATION ET DESSIN DE LA MATRICE
# ==========================================
def evaluate_and_plot(model_name, architecture, val_loader, class_names, num_classes):
    model_path = MODEL_DIR / f"best_{model_name}.pth"

    if not model_path.exists():
        print(f"⚠️ Fichier introuvable: {model_path}. Expérience ignorée.")
        return

    print(f"\nÉvaluation du modèle : {model_name}")

    # Charger le modèle
    model = initialize_model(architecture, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # Inférence
    with torch.no_grad():
        with tqdm(val_loader, desc="Inférence", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # Calcul de la matrice de confusion (COMPTES)
    cm = confusion_matrix(all_labels, all_preds)

    # Conversion en POURCENTAGES par vraie classe (normalisation ligne par ligne)
    cm = cm.astype(np.float32)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    # Éviter la division par zéro pour les classes sans échantillons vrais
    row_sums[row_sums == 0] = 1.0
    cm_percent = (cm / row_sums) * 100.0

    # Paramètres d'affichage (PlantVillage a ~38 classes, donc la figure doit être grande)
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, annot_kws={"size": 8})

    plt.ylabel('Vraies Étiquettes', fontsize=14, fontweight='bold')
    plt.xlabel('Prédictions (pourcentage par vraie classe)', fontsize=14, fontweight='bold')
    plt.title(f'Matrice de Confusion (% par vraie classe) : {model_name}', fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    # Sauvegarde de l'image
    plot_path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"✅ Matrice sauvegardée : {plot_path}")

    # Optionnel: Afficher un rapport de classification rapide dans le terminal
    # print(classification_report(all_labels, all_preds, target_names=class_names))


# ==========================================
# 5. MAIN
# ==========================================
def main():
    print(f"Dispositif utilisé : {device}")

    val_loader, class_names, num_classes = get_val_dataloader_and_classes(batch_size=32)
    print(f"Nombre de classes détectées : {num_classes}")

    # Définissez ici les modèles que vous voulez évaluer.
    # Assurez-vous que les "name" correspondent exactement aux noms de vos expériences dans l'entraînement.
    experiments_to_evaluate = [
        # {"name": "exp1_resnet50_scheduler", "arch": "resnet50"},
        {"name": "exp2_resnet50_class_weights", "arch": "resnet50"},
        {"name": "exp3_resnet50_low_lr", "arch": "resnet50"},
        {"name": "exp4_mobilenetV2", "arch": "mobilenet_v2"},
        {"name": "exp5_resnet50_SGD", "arch": "resnet50"},
        {"name": "exp6_resnet50_weight_decay", "arch": "resnet50"}
    ]

    for exp in experiments_to_evaluate:
        evaluate_and_plot(
            model_name=exp["name"],
            architecture=exp["arch"],
            val_loader=val_loader,
            class_names=class_names,
            num_classes=num_classes
        )

    print("\n🎉 Toutes les évaluations sont terminées. Vérifiez le dossier 'outputs/plots/'.")


if __name__ == "__main__":
    main()