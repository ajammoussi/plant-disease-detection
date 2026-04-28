# Cross-Dataset Evaluation Workflow

## Overview

This directory contains scripts and utilities for evaluating how well your plant disease detection model generalizes to new datasets. The workflow enables you to test your PlantVillage-trained model against alternative plant disease datasets like PlantDoc.

## Components

### 1. Scripts

#### `scripts/create_class_mapping.py`
**Purpose**: Create intelligent mappings between source and target dataset class names

**What it does**:
- Uses fuzzy string matching to map PlantDoc classes to PlantVillage classes
- Generates confidence scores for each mapping
- Saves mapping to `outputs/class_mapping.json`
- Handles unmapped classes gracefully

**Usage**:
```bash
python scripts/create_class_mapping.py
```

**Output**:
- Console output with all mappings and confidence scores
- `outputs/class_mapping.json` with detailed mapping info

#### `scripts/test_cross_dataset.py`
**Purpose**: Evaluate trained model on alternative dataset using class mappings

**What it does**:
- Loads trained ResNet50 model from PlantVillage
- Loads PlantDoc images using class mapping
- Runs inference on all images
- Generates confusion matrices, metrics, and analysis
- Identifies domain shift and generalization issues

**Usage**:
```bash
python scripts/test_cross_dataset.py
```

**Output**:
All saved to `outputs/evaluation/`:
- `cross_dataset_confusion_matrix.png` - Visual confusion matrix
- `confidence_distribution.png` - Confidence analysis
- `cross_dataset_classification_report.txt` - Metrics report
- `cross_dataset_per_class_metrics.csv` - Per-class breakdown

### 2. Configuration Files

#### `outputs/class_mapping.json`
Contains PlantDoc→PlantVillage class mappings with:
- Target class name
- Confidence score (0-1)
- Top 5 candidate matches with scores

**Example**:
```json
{
  "Tomato leaf bacterial spot": {
    "target_class": "Tomato___Bacterial_spot",
    "confidence": 0.933,
    "candidates": [...]
  }
}
```

### 3. Generated Reports

#### `outputs/evaluation/CROSS_DATASET_ANALYSIS_REPORT.md`
Comprehensive analysis including:
- Executive summary
- Domain shift analysis
- Per-class performance breakdown
- Confidence calibration analysis
- 6 improvement recommendations
- Actionable next steps

#### `outputs/evaluation/cross_dataset_per_class_metrics.csv`
Spreadsheet with per-class metrics:
- PlantDoc class name
- Target PlantVillage class
- Accuracy, Precision, Recall, F1-score
- Support (number of samples)

#### `outputs/evaluation/cross_dataset_classification_report.txt`
Standard sklearn classification report with weighted/macro averages

### 4. Visualizations

#### `outputs/evaluation/cross_dataset_confusion_matrix.png`
- 38×38 heatmap (PlantVillage classes)
- Shows what the model predicts for each class
- Darker diagonal = better performance
- Off-diagonal entries show confusion patterns

#### `outputs/evaluation/confidence_distribution.png`
- Two histograms showing prediction confidence
- Correct vs incorrect predictions
- Overall confidence distribution
- Helps identify miscalibration

## Workflow

### Step 1: Prepare New Dataset
```bash
# Place your dataset in a structure like:
# data/YourDataset/train/
# ├── Class_1/
# │   ├── image1.jpg
# │   └── image2.jpg
# └── Class_2/
#     └── image3.jpg
```

### Step 2: Create Class Mapping
```bash
# Edit scripts/test_cross_dataset.py:
# Change PLANTDOC_DIR to your dataset path
python scripts/create_class_mapping.py
# Review the console output - edit mapping if needed
```

### Step 3: Run Evaluation
```bash
python scripts/test_cross_dataset.py
# This will generate all reports and visualizations
```

### Step 4: Analyze Results
```bash
# Review the generated files:
# 1. outputs/evaluation/CROSS_DATASET_ANALYSIS_REPORT.md
# 2. outputs/evaluation/cross_dataset_confusion_matrix.png
# 3. outputs/evaluation/cross_dataset_per_class_metrics.csv
```

## Customization

### Testing Different Model
In `scripts/test_cross_dataset.py`, change:
```python
PLANTVILLAGE_MODEL = PROJECT_ROOT / "outputs" / "models" / "YOUR_MODEL.pth"
```

### Testing Different Dataset
In `scripts/test_cross_dataset.py`, change:
```python
PLANTDOC_DIR = PROJECT_ROOT / "data" / "YOUR_DATASET" / "train"
```

### Adjusting Mapping Confidence Threshold
In `scripts/create_class_mapping.py`, change:
```python
def find_best_match(..., threshold=0.45):  # Change 0.45 to your threshold
```

## Interpreting Results

### Key Metrics Explained

**Accuracy**: Percentage of correct predictions
- Good: > 70%
- OK: 50-70%
- Poor: < 50%

**Precision**: Of predicted positive, how many are actually positive
- High precision = fewer false positives

**Recall**: Of actual positives, how many did we find
- High recall = fewer false negatives

**F1-Score**: Harmonic mean of precision and recall
- Balanced metric: 0 (worst) to 1 (best)

**Confidence-Accuracy Gap**:
- If High confidence + Low accuracy → Model overfitting
- If High confidence + High accuracy → Good generalization
- If Low confidence + Low accuracy → Model uncertainty

### Domain Shift Indicators

1. **High confidence + Low accuracy**: Model memorized dataset-specific features
2. **Class-specific poor performance**: Some diseases don't transfer well
3. **Systematic confusion pattern**: Confusion matrix shows patterns beyond random
4. **Confidence threshold effect**: Accuracy improves significantly with higher confidence

## Troubleshooting

### Script fails to run
```bash
# Check dependencies
pip install torch torchvision sklearn matplotlib seaborn pillow tqdm

# Check dataset exists
ls data/PlantDoc/train/
```

### Low/missing class mappings
```bash
# Check PLANTVILLAGE_DIR path is correct
# Adjust fuzzy matching threshold in create_class_mapping.py
# Manually edit class_mapping.json if needed
```

### Out of memory
```bash
# Reduce batch size in test_cross_dataset.py
# Change: DataLoader(dataset, batch_size=32, ...)
# To: DataLoader(dataset, batch_size=16, ...)
```

### GPU not being used
```bash
# Check device detection
python -c "import torch; print(torch.cuda.is_available())"

# The script auto-selects CUDA/MPS/CPU
```

## Example Output Summary

```
CROSS-DATASET EVALUATION RESULTS
═════════════════════════════════

Dataset: PlantDoc (2,234 images, 27 classes)
Model: ResNet50 (trained on PlantVillage)

Overall Accuracy: 19.61%
Mean Confidence: 0.7159
Average F1: 0.121

Best class: Corn Northern Leaf Blight (F1: 0.464)
Worst class: Multiple (F1: 0.000)

Domain Shift: SIGNIFICANT (confidence-accuracy gap: 52%)
Recommendation: Fine-tune model on target dataset
```

## Next Steps for Improvement

See `QUICK_REFERENCE_GUIDE.md` and `outputs/evaluation/CROSS_DATASET_ANALYSIS_REPORT.md` for:
1. Quick fine-tuning script
2. Domain adaptation approach
3. Full retraining strategy
4. Performance expectations for each

## Files Generated by Evaluation

```
outputs/
├── evaluation/
│   ├── CROSS_DATASET_ANALYSIS_REPORT.md       # Detailed analysis
│   ├── cross_dataset_confusion_matrix.png      # Confusion heatmap
│   ├── confidence_distribution.png            # Confidence analysis
│   ├── cross_dataset_classification_report.txt # Metrics
│   └── cross_dataset_per_class_metrics.csv     # Per-class breakdown
├── class_mapping.json                          # Class mappings
└── models/
    └── best_resnet50_plantvillage.pth          # Evaluated model
```

## Questions?

1. **What's a good accuracy?** For agricultural use: 70-85%+
2. **How long does evaluation take?** ~10-15 minutes on GPU
3. **Can I use this for other datasets?** Yes, modify PLANTDOC_DIR
4. **What if my dataset won't map?** Check class names, adjust threshold
5. **How do I improve accuracy?** See CROSS_DATASET_ANALYSIS_REPORT.md

---

**Last Updated**: April 28, 2026  
**Status**: ✅ Ready for cross-dataset evaluation

