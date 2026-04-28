"""
Create a mapping between PlantDoc and PlantVillage class names.
Uses fuzzy string matching to suggest mappings.
"""

import json
from pathlib import Path
from difflib import SequenceMatcher

# Define datasets paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANTVILLAGE_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage dataset" / "color"
PLANTDOC_DIR = PROJECT_ROOT / "data" / "PlantDoc" / "train"
MAPPING_FILE = PROJECT_ROOT / "outputs" / "class_mapping.json"

def get_all_classes(directory):
    """Get all class directories from a dataset."""
    if not directory.exists():
        return []
    return sorted([d.name for d in directory.iterdir() if d.is_dir()])

def extract_plant_and_condition(plantvillage_class):
    """Extract plant type and condition from PlantVillage class name."""
    # Format: Plant___Disease or Plant___healthy
    parts = plantvillage_class.split("___")
    if len(parts) == 2:
        plant = parts[0].replace("_", " ").lower()
        condition = parts[1].replace("_", " ").lower()
        return plant, condition
    return None, None

def similarity_score(s1, s2):
    """Calculate similarity score between two strings (0-1)."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def find_best_match(plantdoc_class, plantvillage_classes, threshold=0.5):
    """
    Find the best matching PlantVillage class for a PlantDoc class.
    Returns tuple: (matched_class, score, all_candidates_by_score)
    """
    scores = []

    for pv_class in plantvillage_classes:
        # Direct similarity
        direct_score = similarity_score(plantdoc_class, pv_class)

        # Extract components from PlantVillage class
        plant, condition = extract_plant_and_condition(pv_class)

        # Check for plant match in PlantDoc class
        plant_score = 0
        if plant and plant in plantdoc_class.lower():
            plant_score = 0.7

        # Check for disease match in PlantDoc class
        condition_score = 0
        if condition and condition in plantdoc_class.lower():
            condition_score = 0.7

        # Combined score (weighted)
        combined_score = max(direct_score, (plant_score + condition_score) / 2 + direct_score * 0.3)

        scores.append({
            'class': pv_class,
            'score': combined_score,
            'direct_score': direct_score,
            'plant_score': plant_score,
            'condition_score': condition_score
        })

    # Sort by score
    scores.sort(key=lambda x: x['score'], reverse=True)

    best = scores[0] if scores else None

    if best and best['score'] >= threshold:
        return best['class'], best['score'], scores[:5]
    else:
        return None, 0, scores[:5]

def create_mapping():
    """Create comprehensive mapping between PlantDoc and PlantVillage classes."""
    pv_classes = get_all_classes(PLANTVILLAGE_DIR)
    pd_classes = get_all_classes(PLANTDOC_DIR)

    print(f"PlantVillage classes: {len(pv_classes)}")
    print(f"PlantDoc classes: {len(pd_classes)}")
    print()

    mapping = {}
    unmapped = []

    for pd_class in pd_classes:
        best_match, score, candidates = find_best_match(pd_class, pv_classes, threshold=0.45)

        if best_match:
            mapping[pd_class] = {
                'target_class': best_match,
                'confidence': float(score),
                'candidates': [
                    {
                        'class': c['class'],
                        'score': float(c['score']),
                        'combined': f"direct={float(c['direct_score']):.3f}, plant={float(c['plant_score']):.3f}, cond={float(c['condition_score']):.3f}"
                    }
                    for c in candidates
                ]
            }
            status = "✓" if score >= 0.6 else "⚠"
            print(f"{status} {pd_class:50} -> {best_match:50} (score: {score:.3f})")
        else:
            unmapped.append(pd_class)
            print(f"✗ {pd_class:50} -> UNMAPPED (best score: {score:.3f})")
            if candidates:
                for c in candidates[:3]:
                    print(f"    → {c['class']:45} (score: {c['score']:.3f})")

    print(f"\n\nSummary:")
    print(f"  Mapped: {len(mapping)}/{len(pd_classes)}")
    print(f"  Unmapped: {len(unmapped)}/{len(pd_classes)}")

    if unmapped:
        print(f"\nUnmapped classes:")
        for cls in unmapped:
            print(f"  - {cls}")

    # Save mapping
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✅ Mapping saved to: {MAPPING_FILE}")
    return mapping

if __name__ == "__main__":
    create_mapping()

