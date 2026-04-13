import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
DATA_DIR = Path("Data/lung_colon_image_set")
OUTPUT_DIR = Path("data_prepared")
IMG_SIZE = (224, 224)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# Class labels
CLASSES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

def create_directory_structure(base_path):
    """Create train/val/test directory structure for each class"""
    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            dir_path = base_path / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory structure created in {base_path}")

def collect_all_images():
    """Collect images from Train and Validation Set (respecting existing Test Set)"""
    print("📥 Collecting images from source directories...")
    train_val_images = {}
    test_images = {}
    
    for class_name in CLASSES:
        train_val_images[class_name] = []
        test_images[class_name] = []
        
        # Get images from "Train and Validation Set" (will be split into train/val)
        train_val_path = DATA_DIR / "Train and Validation Set" / class_name
        if train_val_path.exists():
            images = list(train_val_path.glob("*.jpg")) + list(train_val_path.glob("*.jpeg")) + list(train_val_path.glob("*.png"))
            train_val_images[class_name].extend(images)
            print(f"  Train/Val for {class_name}: {len(images)} images")
        
        # Get images from existing "Test Set" (will be used as-is)
        test_path = DATA_DIR / "Test Set" / class_name
        if test_path.exists():
            images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.jpeg")) + list(test_path.glob("*.png"))
            test_images[class_name].extend(images)
            print(f"  Existing Test Set for {class_name}: {len(images)} images")
        
        print(f"  Total for {class_name}: {len(train_val_images[class_name]) + len(test_images[class_name])} images")
    
    return train_val_images, test_images

def preprocess_image(img_path, output_path):
    """Read, resize to 224x224, and normalize image"""
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️ Failed to read {img_path}")
            return False
        
        # Resize to 224x224
        img = cv2.resize(img, IMG_SIZE)
        
        # Convert BGR to RGB (for consistency)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert back to uint8 for saving as image
        img = (img * 255).astype(np.uint8)
        
        # Convert back to BGR for cv2.imwrite
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(str(output_path), img)
        return True
    except Exception as e:
        print(f"  ⚠️ Error processing {img_path}: {e}")
        return False

def split_and_organize_data(train_val_images, test_images):
    """
    Split train/val images into train and val.
    Use existing test images as-is.
    """
    print("\n🔄 Splitting data and organizing into train/val/test...")
    
    for class_name in CLASSES:
        train_val_list = [str(img) for img in train_val_images[class_name]]
        test_list = [str(img) for img in test_images[class_name]]
        
        if len(train_val_list) == 0:
            print(f"  ⚠️ No train/val images found for {class_name}")
            continue
        
        # Split train+val into train (70% of original data) and val (15% of original data)
        # From the 85% that's train+val: train should be 70/85 ≈ 0.824, val should be 15/85 ≈ 0.176
        train, val = train_test_split(
            train_val_list,
            test_size=0.1765,  # This gives us 15% of original from val
            random_state=RANDOM_STATE
        )
        
        print(f"\n  {class_name}:")
        print(f"    Train: {len(train)} | Val: {len(val)} | Test (existing): {len(test_list)}")
        
        # Process train images
        train_dir = OUTPUT_DIR / "train" / class_name
        for img_path in tqdm(train, desc=f"    Processing train", leave=False):
            filename = Path(img_path).stem + ".jpg"
            output_path = train_dir / filename
            preprocess_image(img_path, output_path)
        
        # Process val images
        val_dir = OUTPUT_DIR / "val" / class_name
        for img_path in tqdm(val, desc=f"    Processing val", leave=False):
            filename = Path(img_path).stem + ".jpg"
            output_path = val_dir / filename
            preprocess_image(img_path, output_path)
        
        # Process existing test images (from Test Set folder)
        test_dir = OUTPUT_DIR / "test" / class_name
        for img_path in tqdm(test_list, desc=f"    Processing test", leave=False):
            filename = Path(img_path).stem + ".jpg"
            output_path = test_dir / filename
            preprocess_image(img_path, output_path)
        
        print(f"    ✅ train: {len(list(train_dir.glob('*.*')))} images")
        print(f"    ✅ val: {len(list(val_dir.glob('*.*')))} images")
        print(f"    ✅ test (from Test Set): {len(list(test_dir.glob('*.*')))} images")

def print_summary():
    """Print dataset summary"""
    print("\n" + "="*70)
    print("📊 DATASET SUMMARY")
    print("="*70)
    
    for split in ["train", "val", "test"]:
        total = 0
        split_dir = OUTPUT_DIR / split
        
        print(f"\n{split.upper()} Set:")
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            count = len(list(class_dir.glob("*.*")))
            total += count
            print(f"  {class_name:15} : {count:4} images")
        print(f"  {'TOTAL':15} : {total:4} images")

def main():
    print("="*70)
    print("🚀 LC25000 DATASET PREPARATION")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"Train/Val split ratio: Train≈70% | Val≈15%")
    print(f"Test: Using existing Test Set folder (respected as-is)")
    
    # Create directory structure
    create_directory_structure(OUTPUT_DIR)
    
    # Collect images from both sources
    train_val_images, test_images = collect_all_images()
    
    # Split and organize (respecting existing Test Set)
    split_and_organize_data(train_val_images, test_images)
    
    # Print summary
    print_summary()
    
    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n📂 Ready for training at: {OUTPUT_DIR.absolute()}")
    print("✓ Test Set from Data/lung_colon_image_set/Test Set was respected")
    print("\nNext step: Run training script with the prepared dataset")

if __name__ == "__main__":
    main()
