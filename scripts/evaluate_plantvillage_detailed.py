"""
Detailed evaluation of PlantVillage model with error analysis and confidence scores.
Shows precision, recall, F1 for each class and analyzes confidence of misclassified images.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage dataset" / "color"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}\n")


# ==========================================
# DATA LOADING
# ==========================================
def get_val_dataloader_and_classes(batch_size=32, val_split=0.2):
    """Load validation data and class names."""
    print("Loading PlantVillage validation data...")

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

    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return val_loader, class_names, num_classes, val_dataset


# ==========================================
# MODEL INITIALIZATION
# ==========================================
def initialize_model(num_classes):
    """Load trained ResNet50 model."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model_path = MODEL_DIR / "best_resnet50_plantvillage.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded trained model from {model_path}\n")
    else:
        print(f"✗ Model file not found: {model_path}\n")

    return model.to(device)


# ==========================================
# DETAILED EVALUATION WITH CONFIDENCE ANALYSIS
# ==========================================
def detailed_error_analysis():
    """Evaluate model on PlantVillage and analyze errors."""

    print("="*70)
    print("PLANTVILLAGE VALIDATION EVALUATION - DETAILED ERROR ANALYSIS")
    print("="*70 + "\n")

    # Load data
    val_loader, class_names, num_classes, val_dataset = get_val_dataloader_and_classes(batch_size=32)
    print(f"Validation set size: {len(val_dataset)} images")
    print(f"Number of classes: {num_classes}\n")

    # Load model
    model = initialize_model(num_classes)
    model.eval()

    # Run inference and collect detailed data
    print("Running inference...")
    all_preds = []
    all_labels = []
    all_confidences = []
    all_pred_probs = []
    error_data = []  # Store details of misclassifications

    with torch.no_grad():
        with tqdm(val_loader, desc="Inference", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confidences, preds = torch.max(probs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_pred_probs.extend(probs.cpu().numpy())

                # Collect misclassification data
                for i in range(len(labels)):
                    pred = preds[i].item()
                    label = labels[i].item()
                    confidence = confidences[i].item()

                    if pred != label:  # Misclassification
                        # Get top 3 predictions for this image
                        top3_probs, top3_classes = torch.topk(probs[i], 3)
                        error_data.append({
                            'true_class': class_names[label],
                            'pred_class': class_names[pred],
                            'confidence': confidence,
                            'top3_classes': [class_names[c] for c in top3_classes.cpu().numpy()],
                            'top3_probs': [f"{p:.4f}" for p in top3_probs.cpu().numpy()]
                        })

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_pred_probs = np.array(all_pred_probs)

    # ==========================================
    # OVERALL METRICS
    # ==========================================
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE")
    print("="*70)

    overall_accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Total images: {len(all_labels)}")
    print(f"Correct predictions: {np.sum(all_preds == all_labels)}")
    print(f"Incorrect predictions: {np.sum(all_preds != all_labels)}")

    # Confidence statistics
    correct_mask = (all_preds == all_labels)
    print(f"\nConfidence of Correct Predictions:")
    print(f"  Mean: {all_confidences[correct_mask].mean():.4f}")
    print(f"  Std:  {all_confidences[correct_mask].std():.4f}")
    print(f"  Min:  {all_confidences[correct_mask].min():.4f}")
    print(f"  Max:  {all_confidences[correct_mask].max():.4f}")

    incorrect_mask = (all_preds != all_labels)
    if np.sum(incorrect_mask) > 0:
        print(f"\nConfidence of Incorrect Predictions:")
        print(f"  Mean: {all_confidences[incorrect_mask].mean():.4f}")
        print(f"  Std:  {all_confidences[incorrect_mask].std():.4f}")
        print(f"  Min:  {all_confidences[incorrect_mask].min():.4f}")
        print(f"  Max:  {all_confidences[incorrect_mask].max():.4f}")

    # ==========================================
    # PER-CLASS METRICS
    # ==========================================
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE (SORTED BY F1-SCORE)")
    print("="*70)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), average=None, zero_division=0
    )

    # Create dataframe for easy viewing
    class_metrics = []
    for idx, class_name in enumerate(class_names):
        class_metrics.append({
            'class': class_name,
            'precision': precision[idx],
            'recall': recall[idx],
            'f1': f1[idx],
            'support': int(support[idx]),
            'accuracy': np.sum((all_labels == idx) & (all_preds == idx)) / max(np.sum(all_labels == idx), 1)
        })

    # Sort by F1 score descending
    class_metrics.sort(key=lambda x: x['f1'], reverse=True)

    print(f"\n{'Class Name':<50} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Acc':<8} {'Supp':<6}")
    print("-" * 130)

    for item in class_metrics:
        print(f"{item['class']:<50} {item['precision']:<8.4f} {item['recall']:<8.4f} "
              f"{item['f1']:<8.4f} {item['accuracy']:<8.4f} {item['support']:<6d}")

    # ==========================================
    # HIGHLIGHT NEAR-PERFECT CLASSES
    # ==========================================
    print("\n" + "="*70)
    print("NEAR-PERFECT CLASSES (F1 ≥ 0.95)")
    print("="*70)

    near_perfect = [m for m in class_metrics if m['f1'] >= 0.95]

    if near_perfect:
        print(f"\nFound {len(near_perfect)} near-perfect classes:\n")
        for item in near_perfect:
            print(f"✓ {item['class']}")
            print(f"  Precision: {item['precision']:.4f}")
            print(f"  Recall:    {item['recall']:.4f}")
            print(f"  F1-Score:  {item['f1']:.4f}")
            print(f"  Accuracy:  {item['accuracy']:.4f}")
            print(f"  Support:   {item['support']} samples")

            # Errors for this class
            class_idx = class_names.index(item['class'])
            class_errors = [e for e in error_data if e['true_class'] == item['class']]

            if class_errors:
                print(f"  Errors: {len(class_errors)} misclassifications")
                print(f"  Error confidence - Mean: {np.mean([e['confidence'] for e in class_errors]):.4f}")
                print(f"  Top confusion:")
                confusion_counts = defaultdict(int)
                for e in class_errors:
                    confusion_counts[e['pred_class']] += 1
                for pred_class, count in sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    → Confused with {pred_class}: {count} times")
            else:
                print(f"  Errors: None - Perfect classification!")
            print()
    else:
        print("\nNo near-perfect classes (F1 ≥ 0.95)")

    # ==========================================
    # ERROR ANALYSIS
    # ==========================================
    if error_data:
        print("="*70)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*70)

        print(f"\nTotal misclassifications: {len(error_data)}")
        print(f"Error rate: {len(error_data)/len(all_labels)*100:.2f}%")

        # Confidence of errors
        error_confidences = [e['confidence'] for e in error_data]
        print(f"\nConfidence of Errors:")
        print(f"  Mean: {np.mean(error_confidences):.4f}")
        print(f"  Std:  {np.std(error_confidences):.4f}")
        print(f"  Min:  {np.min(error_confidences):.4f}")
        print(f"  Max:  {np.max(error_confidences):.4f}")

        # Confidence distribution of errors
        print(f"\nError Confidence Distribution:")
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        for i in range(len(bins)-1):
            count = sum(1 for c in error_confidences if bins[i] <= c < bins[i+1])
            pct = count / len(error_confidences) * 100
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {count:3d} ({pct:5.1f}%)")

        # Save error samples
        print(f"\n" + "-"*70)
        print("Sample Misclassifications (first 20):")
        print("-"*70)

        for i, error in enumerate(error_data[:20]):
            print(f"\n{i+1}. True: {error['true_class']}")
            print(f"   Pred: {error['pred_class']} (confidence: {error['confidence']:.4f})")
            print(f"   Top 3 predictions:")
            for j, (cls, prob) in enumerate(zip(error['top3_classes'], error['top3_probs'])):
                print(f"     {j+1}. {cls}: {prob}")

    # ==========================================
    # SAVE DETAILED REPORT TO CSV
    # ==========================================
    print("\n" + "="*70)
    print("SAVING DETAILED REPORTS")
    print("="*70)

    # Save per-class metrics
    df_class_metrics = pd.DataFrame(class_metrics)
    csv_path = EVAL_DIR / "plantvillage_per_class_metrics_detailed.csv"
    df_class_metrics.to_csv(csv_path, index=False)
    print(f"✓ Per-class metrics saved: {csv_path}")

    # Save error data
    if error_data:
        df_errors = pd.DataFrame(error_data)
        error_path = EVAL_DIR / "plantvillage_misclassifications.csv"
        df_errors.to_csv(error_path, index=False)
        print(f"✓ Misclassifications saved: {error_path}")

    # Save classification report
    class_report = classification_report(
        all_labels, all_preds,
        labels=range(num_classes),
        target_names=class_names,
        zero_division=0
    )

    report_path = EVAL_DIR / "plantvillage_classification_report_detailed.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PLANTVILLAGE VALIDATION EVALUATION - DETAILED REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n\n")
        f.write(f"Total Images: {len(all_labels)}\n")
        f.write(f"Correct: {np.sum(all_preds == all_labels)}\n")
        f.write(f"Incorrect: {np.sum(all_preds != all_labels)}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-"*80 + "\n")
        f.write(class_report)

    print(f"✓ Classification report saved: {report_path}")

    # ==========================================
    # VISUALIZATION: CONFIDENCE BY CORRECTNESS
    # ==========================================
    print(f"\n✓ Generating confidence distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Correct vs Incorrect
    axes[0].hist(all_confidences[correct_mask], bins=50, alpha=0.7, label='Correct', color='green', density=True)
    if np.sum(incorrect_mask) > 0:
        axes[0].hist(all_confidences[incorrect_mask], bins=50, alpha=0.7, label='Incorrect', color='red', density=True)
    axes[0].set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Overall with mean line
    axes[1].hist(all_confidences, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[1].axvline(all_confidences.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {all_confidences.mean():.3f}')
    axes[1].set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Overall Confidence Distribution (PlantVillage)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = EVAL_DIR / "plantvillage_confidence_distribution.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confidence plot saved: {viz_path}")

    # ==========================================
    # CONFUSION MATRIX FOR ERRORS
    # ==========================================
    print(f"✓ Generating confusion matrix...")

    cm = confusion_matrix(all_labels, all_preds)

    # Normalize to percentage
    cm_percent = cm.astype(np.float32)
    row_sums = cm_percent.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1.0
    cm_percent = (cm_percent / row_sums) * 100.0

    plt.figure(figsize=(24, 20))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, annot_kws={"size": 7})

    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix (%) - PlantVillage Validation', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = EVAL_DIR / "plantvillage_confusion_matrix_detailed.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {cm_path}")

    print("\n" + "="*70)
    print("✅ DETAILED EVALUATION COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {EVAL_DIR}")
    print("\nGenerated files:")
    print(f"  - {csv_path.name}")
    print(f"  - {report_path.name}")
    if error_data:
        print(f"  - {error_path.name}")
    print(f"  - {viz_path.name}")
    print(f"  - {cm_path.name}")


if __name__ == "__main__":
    detailed_error_analysis()

