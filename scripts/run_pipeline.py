import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm  # <-- Added tqdm for progress tracking

# Add the project root to sys.path so 'src' is importable
# Assuming this script is inside a 'scripts' folder at the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your custom modules
from src import data_loader, data_cleaning, preprocessing

def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Create output directory for processed images
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_SIZE = (256, 256)
    
    print("=== Plant Disease Detection Preprocessing Pipeline ===")
    
    # ---------------------------------------------------------
    # 2. Scanning Dataset
    # ---------------------------------------------------------
    print("\nScanning dataset...")
    dataset_root = data_loader.find_dataset_root(str(RAW_DIR))
    class_map = data_loader.scan_dataset(str(dataset_root))
    
    # Isolate original images from pre-segmented ones
    class_map, _ = data_loader.split_segmented_originals(class_map)
    
    # ---------------------------------------------------------
    # 3. Data Cleaning (In-memory mapping)
    # ---------------------------------------------------------
    print("Cleaning dataset...")
    
    # Detect and remove corrupt images from the processing map
    corrupt_report = data_cleaning.find_corrupt_images(class_map, verbose=False)
    class_map = data_cleaning.remove_corrupt_images(
        corrupt_report=corrupt_report, 
        class_map=class_map, 
        dry_run=True  # Keeps raw files safe, removes from memory
    )
    
    # Detect and remove exact duplicates from the processing map
    duplicate_groups = data_cleaning.find_exact_duplicates(class_map)
    class_map = data_cleaning.remove_duplicates(
        class_map=class_map, 
        duplicate_groups=duplicate_groups, 
        dry_run=True
    )
    
    total_images = sum(len(paths) for paths in class_map.values())
    print(f"Total images to process after cleaning: {total_images:,}")
    
    # ---------------------------------------------------------
    # 4. Processing & Saving Images with Progress Bar
    # ---------------------------------------------------------
    print("\nStarting batch preprocessing...")
    
    # Initialize the global progress bar
    with tqdm(total=total_images, desc="Overall Progress", unit="img") as pbar:
        
        for cls_name, paths in class_map.items():
            # Update the progress bar description to show the current class being processed
            pbar.set_postfix_str(f"Current class: {cls_name}")
            
            # Create class directory in the processed folder
            cls_out_dir = PROCESSED_DIR / cls_name
            cls_out_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in paths:
                try:
                    # Load Image
                    img = data_loader.load_image(str(img_path))
                    
                    # Resize (using 'stretch' mode targeting 256x256 as chosen in EDA)
                    img_resized = preprocessing.resize_image(img, TARGET_SIZE, mode="stretch")
                    
                    # Full Enhancement Pipeline (Bilateral Denoise + CLAHE)
                    img_enhanced = preprocessing.full_enhancement_pipeline(img_resized)
                    
                    # Save as a standard enhanced image
                    out_path = cls_out_dir / img_path.name
                    Image.fromarray(img_enhanced).save(out_path)
                    
                except Exception as e:
                    # Use tqdm.write instead of print so it doesn't break the progress bar layout
                    tqdm.write(f"  [ERROR] Failed to process {img_path.name} in {cls_name}: {e}")
                
                finally:
                    # Increment the progress bar by 1 for every image processed
                    pbar.update(1)

    print("\n✅ Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()