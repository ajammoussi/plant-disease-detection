import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_model(model_path, num_classes, device):
    """Loads the ResNet50 model and injects the trained weights."""
    print("Loading model architecture and weights...")
    
    # Initialize a blank ResNet50
    model = models.resnet50(weights=None)
    
    # Replace the final layer to match our 38 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Set model to evaluation mode (turns off dropout, batchnorm updates, etc.)
    model.eval()
    return model

def predict_image(image_path, model, class_names, device):
    """Processes an image and returns the top 3 predictions."""
    # 1. Define the exact same validation transform
    transform = transforms.Compose([
        transforms.Resize(256),        # Resize shortest edge to 256
        transforms.CenterCrop(224),    # Crop the center 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 2. Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Could not load image at {image_path}: {e}")
        sys.exit(1)
        
    img_t = transform(img)
    # Add a batch dimension: [C, H, W] -> [1, C, H, W]
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    
# 3. Run Inference
    with torch.no_grad(): 
        outputs = model(batch_t)
        
        # Convert raw logits to probabilities using Softmax 
        # and use .squeeze() to flatten it from to just
        probabilities = F.softmax(outputs, dim=1).squeeze()
        
    # 4. Get Top 3 Predictions
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    print(f"\n--- Analysis for: {Path(image_path).name} ---")
    for i in range(3):  # We know we want the top 3
        score = top3_prob[i].item() * 100
        class_name = class_names[top3_catid[i].item()]
        print(f"{i+1}. {class_name} ({score:.2f}% confidence)")

def main():
    # Setup Argument Parser to accept image paths from the terminal
    parser = argparse.ArgumentParser(description="Test the Plant Disease ResNet50 Model")
    parser.add_argument("image_path", type=str, help="Path to the image you want to test")
    args = parser.parse_args()

    # Paths
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_resnet50_plantvillage.pth"
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}. Did you run train.py first?")
        sys.exit(1)

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Get class names (this reads the folder names from data/processed)
    # This ensures the IDs perfectly match how the DataLoader mapped them during training
    dataset = datasets.ImageFolder(str(PROCESSED_DIR))
    class_names = dataset.classes 
    NUM_CLASSES = len(class_names)
    
    # Load Model and Predict
    model = load_model(MODEL_PATH, NUM_CLASSES, device)
    predict_image(args.image_path, model, class_names, device)

if __name__ == "__main__":
    main()