"""
Cross-dataset evaluation: Test PlantVillage-trained model on PlantDoc dataset.
Evaluates generalization performance and domain shift analysis.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLANTVILLAGE_MODEL = PROJECT_ROOT / "outputs" / "models" / "best_resnet50_plantvillage.pth"
PLANTDOC_DIR = PROJECT_ROOT / "data" / "PlantDoc" / "train"
CLASS_MAPPING_FILE = PROJECT_ROOT / "outputs" / "class_mapping.json"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")


# ==========================================
# 2. LOAD CLASS MAPPING
# ==========================================
def load_mapping():
    """Load PlantDoc-to-PlantVillage class mapping."""
    with open(CLASS_MAPPING_FILE, 'r') as f:
        mapping = json.load(f)
    return mapping


# ==========================================
# 3. BUILD PLANTVILLAGE CLASS LIST
# ==========================================
def get_plantvillage_classes():
    """Get PlantVillage class names in the same order as training."""
    pv_dir = PROJECT_ROOT / "data" / "raw" / "plantvillage dataset" / "color"
    classes = sorted([d.name for d in pv_dir.iterdir() if d.is_dir()])
    return classes


# ==========================================
# 4. CUSTOM DATASET FOR PLANTDOC
# ==========================================
class PlantDocDataset(Dataset):
    """Dataset for PlantDoc images with automatic class mapping."""

    def __init__(self, root_dir, class_mapping, plantvillage_classes, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_mapping = class_mapping
        self.pv_classes = plantvillage_classes
        self.pv_to_idx = {cls: idx for idx, cls in enumerate(plantvillage_classes)}

        self.images = []
        self.labels = []
        self.pd_classes = []
        self.unmapped_count = 0

        # Load all images
        for pd_class in sorted(self.root_dir.iterdir()):
            if not pd_class.is_dir():
                continue

            pd_class_name = pd_class.name
            pv_class = class_mapping.get(pd_class_name, {}).get('target_class')

            if not pv_class:
                self.unmapped_count += len(list(pd_class.glob('*.jpg')))
                continue

            pv_idx = self.pv_to_idx.get(pv_class)
            if pv_idx is None:
                continue

            for img_path in pd_class.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(pv_idx)
                self.pd_classes.append(pd_class_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        pd_class = self.pd_classes[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image
            img = torch.zeros(3, 224, 224)

        return img, label, pd_class


# ==========================================
# 5. MODEL INITIALIZATION
# ==========================================
def initialize_model(num_classes):
    """Load pre-trained ResNet50 model with PlantVillage weights."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load trained weights
    if PLANTVILLAGE_MODEL.exists():
        model.load_state_dict(torch.load(PLANTVILLAGE_MODEL, map_location=device))
        print(f"✓ Loaded trained model from {PLANTVILLAGE_MODEL}")
    else:
        print(f"✗ Model file not found: {PLANTVILLAGE_MODEL}")

    return model.to(device)


# ==========================================
# 6. EVALUATION
# ==========================================
def evaluate_cross_dataset():
    """Evaluate model on PlantDoc dataset and generate report."""

    print("\n" + "="*60)
    print("CROSS-DATASET EVALUATION: PlantVillage Model on PlantDoc Data")
    print("="*60 + "\n")

    # Load mapping and classes
    mapping = load_mapping()
    pv_classes = get_plantvillage_classes()
    num_classes = len(pv_classes)

    print(f"PlantVillage classes: {num_classes}")
    print(f"PlantDoc classes to map: {len(mapping)}")

    # Prepare dataset
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PlantDocDataset(PLANTDOC_DIR, mapping, pv_classes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nDataset statistics:")
    print(f"  Total images loaded: {len(dataset)}")
    print(f"  Unmapped images: {dataset.unmapped_count}")
    print(f"  Images per class (approx):")

    # Count images per mapped class
    class_counts = defaultdict(int)
    for _, _, pd_class in dataset:
        class_counts[pd_class] += 1

    for pd_class in sorted(class_counts.keys()):
        mapped_pv = mapping[pd_class]['target_class']
        confidence = mapping[pd_class]['confidence']
        count = class_counts[pd_class]
        status = "✓" if confidence >= 0.6 else "⚠"
        print(f"    {status} {pd_class:40} -> {mapped_pv:50} ({count} imgs, conf={confidence:.3f})")

    # Load model
    print("\n" + "-"*60)
    model = initialize_model(num_classes)
    model.eval()

    # Run inference
    print("\nRunning inference...")
    all_preds = []
    all_labels = []
    all_confidences = []
    all_pd_classes = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Inference", unit="batch") as pbar:
            for inputs, labels, pd_classes in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confidences, preds = torch.max(probs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_pd_classes.extend(pd_classes)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # ==========================================
    # 7. GENERATE METRICS
    # ==========================================
    print("\n" + "-"*60)
    print("EVALUATION METRICS")
    print("-"*60)

    # Overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

    # Confidence statistics
    print(f"\nPrediction Confidence:")
    print(f"  Mean: {all_confidences.mean():.4f}")
    print(f"  Std:  {all_confidences.std():.4f}")
    print(f"  Min:  {all_confidences.min():.4f}")
    print(f"  Max:  {all_confidences.max():.4f}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), average=None, zero_division=0
    )
    
    print(f"\nPer-class Performance (sorted by F1-score):")
    print(f"{'PlantDoc Class':<40} {'PV Class':<40} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-" * 110)
    
    class_performance = []
    for idx, pv_class in enumerate(pv_classes):
        # Find corresponding PlantDoc class(es)
        pd_classes_for_pv = [pd_cls for pd_cls, mapping_info in mapping.items() 
                             if mapping_info.get('target_class') == pv_class]
        
        pd_class_str = ", ".join(pd_classes_for_pv) if pd_classes_for_pv else "N/A"
        
        acc = np.sum((all_labels == idx) & (all_preds == idx)) / max(np.sum(all_labels == idx), 1)
        
        if idx < len(precision):
            class_performance.append({
                'pv_class': pv_class,
                'pd_classes': pd_class_str,
                'accuracy': acc,
                'precision': precision[idx],
                'recall': recall[idx],
                'f1': f1[idx],
                'support': support[idx]
            })

    # Sort by F1
    class_performance.sort(key=lambda x: x['f1'], reverse=True)

    for item in class_performance:
        print(f"{item['pd_classes']:<40} {item['pv_class']:<40} "
              f"{item['accuracy']:.4f}    {item['precision']:.4f}    "
              f"{item['recall']:.4f}    {item['f1']:.4f}")

    # ==========================================
    # 8. CONFUSION MATRIX
    # ==========================================
    print("\n" + "-"*60)
    print("Generating confusion matrix...")

    cm = confusion_matrix(all_labels, all_preds)

    # Convert to percentages (row-wise)
    cm_percent = cm.astype(np.float32)
    row_sums = cm_percent.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1.0
    cm_percent = (cm_percent / row_sums) * 100.0

    # Plot
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=pv_classes, yticklabels=pv_classes,
                cbar=True, annot_kws={"size": 7})

    plt.ylabel('True Label (PlantVillage)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label (PlantVillage)', fontsize=12, fontweight='bold')
    plt.title('Cross-Dataset Confusion Matrix (%)\nPlantVillage Model on PlantDoc Data',
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = EVAL_DIR / "cross_dataset_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {cm_path}")

    # ==========================================
    # 9. CONFIDENCE DISTRIBUTION
    # ==========================================
    print("\nGenerating confidence distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correct vs Incorrect predictions
    correct_mask = all_preds == all_labels
    axes[0].hist(all_confidences[correct_mask], bins=50, alpha=0.7, label='Correct', color='green')
    axes[0].hist(all_confidences[~correct_mask], bins=50, alpha=0.7, label='Incorrect', color='red')
    axes[0].set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Overall histogram
    axes[1].hist(all_confidences, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[1].axvline(all_confidences.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_confidences.mean():.3f}')
    axes[1].set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Overall Prediction Confidence', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    conf_path = EVAL_DIR / "confidence_distribution.png"
    plt.tight_layout()
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confidence plot saved: {conf_path}")

    # ==========================================
    # 10. SAVE DETAILED REPORT
    # ==========================================
    print("\nGenerating detailed report...")

    report_df = pd.DataFrame(class_performance)
    csv_path = EVAL_DIR / "cross_dataset_per_class_metrics.csv"
    report_df.to_csv(csv_path, index=False)
    print(f"✓ CSV report saved: {csv_path}")

    # Classification report
    class_report = classification_report(
        all_labels, all_preds, 
        labels=range(num_classes),
        target_names=pv_classes,
        zero_division=0
    )

    report_path = EVAL_DIR / "cross_dataset_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-DATASET EVALUATION REPORT\n")
        f.write("PlantVillage Model Tested on PlantDoc Dataset\n")
        f.write("="*80 + "\n\n")

        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n\n")

        f.write("Mean Prediction Confidence:\n")
        f.write(f"  Mean: {all_confidences.mean():.4f}\n")
        f.write(f"  Std:  {all_confidences.std():.4f}\n")
        f.write(f"  Min:  {all_confidences.min():.4f}\n")
        f.write(f"  Max:  {all_confidences.max():.4f}\n\n")

        f.write("CLASS-WISE CLASSIFICATION REPORT:\n")
        f.write("-"*80 + "\n")
        f.write(class_report)
        f.write("\n\nPER-CLASS PERFORMANCE DETAILS:\n")
        f.write("-"*80 + "\n")
        f.write(report_df.to_string())

    print(f"✓ Report saved: {report_path}")

    # ==========================================
    # 11. ANALYSIS & INSIGHTS
    # ==========================================
    print("\n" + "="*60)
    print("GENERALIZATION ANALYSIS")
    print("="*60)

    # Low confidence predictions
    low_conf_mask = all_confidences < 0.3
    print(f"\nLow confidence predictions (<0.3): {low_conf_mask.sum()} / {len(all_confidences)} ({low_conf_mask.sum()/len(all_confidences)*100:.2f}%)")

    # Accuracy by confidence threshold
    print("\nAccuracy by confidence threshold:")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        mask = all_confidences >= threshold
        if mask.sum() > 0:
            acc = accuracy_score(all_labels[mask], all_preds[mask])
            print(f"  >= {threshold}: {acc:.4f} ({mask.sum()} samples)")

    # Best and worst performing classes
    print("\nTop 5 Best Performing Classes (by F1):")
    for i, item in enumerate(class_performance[:5]):
        print(f"  {i+1}. {item['pv_class']:45} F1={item['f1']:.4f}")

    print("\nTop 5 Worst Performing Classes (by F1):")
    for i, item in enumerate(class_performance[-5:]):
        print(f"  {i+1}. {item['pv_class']:45} F1={item['f1']:.4f}")

    # Insights
    print("\nKEY INSIGHTS:")
    avg_f1 = np.mean([item['f1'] for item in class_performance])
    print(f"  • Overall average F1-score: {avg_f1:.4f}")
    print(f"  • Model shows {'good' if overall_accuracy > 0.7 else 'moderate' if overall_accuracy > 0.5 else 'limited'} generalization")
    print(f"  • Domain shift detected: {'Significant' if overall_accuracy < 0.5 else 'Moderate' if overall_accuracy < 0.7 else 'Minor'}")

    print("\n✅ Evaluation complete!")
    print(f"All results saved to: {EVAL_DIR}\n")


if __name__ == "__main__":
    evaluate_cross_dataset()



