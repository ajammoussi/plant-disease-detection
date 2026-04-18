import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 38
    VAL_SPLIT = 0.2
    
    # Device configuration (GPU if available, else MPS for Mac, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"=== Plant Disease ResNet50 Fine-Tuning ===")
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # 2. Data Transforms (with ImageNet Normalization)
    # ---------------------------------------------------------
    # Normalization values explicitly from your preprocessing pipeline
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Crop to 224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224), # Center crop the 256x256 image to 224x224
            transforms.ToTensor(),
            normalize
        ]),
    }

    # ---------------------------------------------------------
    # 3. Loading Dataset
    # ---------------------------------------------------------
    print("\nLoading dataset from processed directory...")
    # Load all images using the validation transform first (we will override train later)
    full_dataset = datasets.ImageFolder(str(PROCESSED_DIR))
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply correct transforms
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    
    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    print(f"Training samples: {train_size:,} | Validation samples: {val_size:,}")
    
    # ---------------------------------------------------------
    # 4. Model Definition
    # ---------------------------------------------------------
    print("\nInitializing ResNet50 model...")
    # Load pre-trained ResNet50
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\nStarting training...\n")
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 20)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Use tqdm for batch progress
            with tqdm(dataloaders[phase], desc=phase.capitalize(), unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass & optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")
            
            # Deep copy the model if it's the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_path = MODEL_DIR / "best_resnet50_plantvillage.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"🌟 New best model saved to {best_model_path}!\n")
                
    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()