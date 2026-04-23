import sys
import time
from pathlib import Path
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Configuration des chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage dataset" / "color"
#PROCESSED_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage dataset" / "dummy"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
def get_dataloaders_and_weights(batch_size=32, val_split=0.2):
    print("Chargement et préparation des données...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

    full_dataset = datasets.ImageFolder(str(PROCESSED_DIR))
    
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }

    # Calcul des poids de classes pour gérer le déséquilibre
    print("Calcul des Class Weights...")
    train_indices = train_dataset.indices
    train_targets = [full_dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(train_targets)
    total_samples = len(train_targets)
    num_classes = len(class_counts)
    
    # Formule standard des class weights : N / (C * N_i)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    return dataloaders, num_classes, class_weights_tensor

# ==========================================
# 2. INITIALISATION DU MODÈLE
# ==========================================
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
# 3. FONCTION D'ENTRAÎNEMENT PRINCIPALE
# ==========================================
def train_model(config, dataloaders, num_classes, class_weights_tensor, max_epochs=30, patience=5):
    print(f"\n" + "="*50)
    print(f"🚀 LANCEMENT DE L'EXPÉRIENCE : {config['name']}")
    print(f"Modèle: {config['model']} | LR: {config['lr']} | Poids de classes: {config['use_weights']}")
    print("="*50)

    # 1. Setup modèle, loss, optimizer et logs
    model = initialize_model(config["model"], num_classes)
    
    if config["use_weights"]:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
        
    # Récupérer le weight_decay si défini, sinon 0
    wd = config.get("weight_decay", 0.0)
    
    # Choisir l'optimiseur (Adam par défaut)
    opt_name = config.get("optimizer", "adam")
    
    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=wd)
    
    # Scheduler pour réduire le LR si la perte stagne
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    writer = SummaryWriter(log_dir=str(LOG_DIR / config["name"]))
    best_model_path = MODEL_DIR / f"best_{config['name']}.pth"

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 2. Boucle d'entraînement
    for epoch in range(max_epochs):
        print(f"\nÉpoque {epoch+1}/{max_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            with tqdm(dataloaders[phase], desc=phase.capitalize(), unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Logs pour TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # Logique de Validation : Scheduler et Early Stopping
            if phase == 'val':
                scheduler.step(epoch_loss)

                # Sauvegarde du meilleur modèle basé sur la Loss
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"🌟 Modèle amélioré. Sauvegarde à : {best_model_path}")
                else:
                    epochs_no_improve += 1
                    print(f"⚠️ Pas d'amélioration depuis {epochs_no_improve} époque(s).")

        # Vérification de l'Early Stopping
        if epochs_no_improve >= patience:
            print(f"\n🛑 Early Stopping déclenché à l'époque {epoch+1} ! (Le modèle n'apprend plus)")
            break 

    writer.close()
    print(f"✅ Fin de l'expérience {config['name']}. Meilleure Loss Val: {best_val_loss:.4f}\n")

# ==========================================
# 4. GESTION DES EXPÉRIENCES (Main)
# ==========================================
def main():
    print(f"Dispositif utilisé : {device}")
    
    # 1. Charger les données une seule fois pour toutes les expériences
    dataloaders, num_classes, class_weights_tensor = get_dataloaders_and_weights(batch_size=32)

    # 2. Définir le planning des expériences
    experiments = [
        # Expérience 1: Baseline corrigée (Scheduler + EarlyStopping)
        #{"name": "exp1_resnet50_scheduler", "model": "resnet50", "lr": 1e-4, "use_weights": False},
        
        # Expérience 2: Gestion du déséquilibre des classes
        {"name": "exp2_resnet50_class_weights", "model": "resnet50", "lr": 1e-4, "use_weights": True},
        
        # Expérience 3: Apprentissage lent pour éviter les rebonds
        {"name": "exp3_resnet50_low_lr", "model": "resnet50", "lr": 1e-5, "use_weights": False},
        
        # Expérience 4: Changement d'architecture (Modèle léger)
        {"name": "exp4_mobilenetV2", "model": "mobilenet_v2", "lr": 1e-4, "use_weights": False},
        
        
        # NOUVELLES EXPÉRIENCES (Optionnelles)
        # Exp 5 : SGD au lieu de Adam (On met un LR un peu plus grand car SGD est plus lent)
        {"name": "exp5_resnet50_SGD", "model": "resnet50", "lr": 1e-3, "use_weights": False, "optimizer": "sgd"},
        
        # Exp 6 : Adam mais avec de la régularisation (Weight Decay) pour forcer le modèle à être moins confiant
        {"name": "exp6_resnet50_weight_decay", "model": "resnet50", "lr": 1e-4, "use_weights": False, "weight_decay": 1e-4}
    ]

    # 3. Lancer toutes les expériences automatiquement
    for config in experiments:
        train_model(
            config=config,
            dataloaders=dataloaders,
            num_classes=num_classes,
            class_weights_tensor=class_weights_tensor,
            max_epochs=40,  # Élevé car l'Early Stopping l'arrêtera avant de toute façon
            patience=6      # Arrêt si aucune amélioration après 6 époques
        )

    print("🎉 TOUTES LES EXPÉRIENCES SONT TERMINÉES ! Bonne analyse des résultats.")

if __name__ == "__main__":
    main()
