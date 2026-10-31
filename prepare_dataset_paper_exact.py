"""
Dataset Preparation - EXACT Paper Methodology
Based on: Hessane et al. 2023 - Big Data Mining and Analytics
FIXED: Handles images of different sizes
"""

import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

# Import utilities
from utils.data_preprocessing import ImagePreprocessor, DataAugmenter


class PaperExactDatasetPreparation:
    """
    Exact dataset preparation as described in the paper
    """
    
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Paper specifications
        self.target_size = (224, 224)
        self.train_split = 0.8  # 80% training, 20% testing
        
        # Augmentation parameters from paper (Section 3.2)
        self.augmentation_params = {
            'rotation_range': 40,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True
        }
        
        # Expected class distribution from paper (Table 1)
        self.target_distribution = {
            'Healthy': 1258,
            'Stage1': 1219,
            'Stage2': 1264,
            'Stage3': 1197
        }
    
    def load_raw_images(self):
        """
        Load images from organized folders
        FIXED: Resizes images immediately after loading
        """
        print("=" * 70)
        print("📂 LOADING RAW IMAGES (Paper Methodology)")
        print("=" * 70)
        
        images = []
        labels = []
        filenames = []
        
        class_mapping = {
            'Healthy': 0,
            'Stage1': 1,
            'Stage2': 2,
            'Stage3': 3
        }
        
        for class_name, class_idx in class_mapping.items():
            class_path = self.raw_data_path / class_name
            
            if not class_path.exists():
                print(f"⚠️  Warning: {class_path} not found!")
                continue
            
            # Get all image files
            image_files = list(class_path.glob('*.jpg')) + \
                         list(class_path.glob('*.jpeg')) + \
                         list(class_path.glob('*.png'))
            
            print(f"\n{class_name}: {len(image_files)} images found")
            
            for img_file in image_files:
                try:
                    # Load image
                    img = Image.open(img_file).convert('RGB')
                    
                    # ✅ CRITICAL FIX: Resize immediately
                    img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array (now all same size)
                    img_array = np.array(img)
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    filenames.append(img_file.name)
                    
                except Exception as e:
                    print(f"  ❌ Error loading {img_file.name}: {e}")
        
        print(f"\n✅ Total raw images loaded: {len(images)}")
        
        # Show distribution
        label_counts = Counter(labels)
        class_names = ['Healthy', 'Stage1', 'Stage2', 'Stage3']
        print("\n📊 Initial Distribution:")
        for idx, name in enumerate(class_names):
            count = label_counts.get(idx, 0)
            print(f"   {name}: {count} images")
        
        # Now safe to convert to array (all images are 224x224x3)
        return np.array(images), np.array(labels), filenames
    
    def preprocess_images(self, images):
        """
        Preprocess images (normalize)
        Paper: Section 3.2 - "normalization"
        Images are already resized in load_raw_images
        """
        print("\n" + "=" * 70)
        print("🔧 PREPROCESSING IMAGES")
        print("=" * 70)
        
        print(f"\nNormalizing {len(images)} images to [0, 1] range...")
        
        # Normalize to [0, 1]
        images_normalized = images.astype(np.float32) / 255.0
        
        print(f"✅ All images preprocessed (shape: {images_normalized.shape})")
        
        return images_normalized
    
    def split_train_test(self, images, labels):
        """
        Split into training and testing sets
        Paper: "80% training, 20% testing" (Table 1)
        """
        print("\n" + "=" * 70)
        print(f"✂️  SPLITTING DATASET (Train: 80%, Test: 20%)")
        print("=" * 70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels,
            test_size=0.2,
            stratify=labels,
            random_state=42  # For reproducibility
        )
        
        print(f"\n📊 Split Results:")
        print(f"   Training set: {len(X_train)} images")
        print(f"   Testing set: {len(X_test)} images")
        
        # Show class distribution in splits
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)
        class_names = ['Healthy', 'Stage1', 'Stage2', 'Stage3']
        
        print(f"\n   Training Distribution:")
        for idx, name in enumerate(class_names):
            print(f"      {name}: {train_counts.get(idx, 0)}")
        
        print(f"\n   Testing Distribution:")
        for idx, name in enumerate(class_names):
            print(f"      {name}: {test_counts.get(idx, 0)}")
        
        return X_train, X_test, y_train, y_test
    
    def augment_training_data(self, X_train, y_train):
        """
        Apply data augmentation to training set
        Paper: Section 3.2 - "zooming, rotation, horizontal flipping, 
               width and height shifting"
        
        Target: Match paper's final counts (Table 1)
        """
        print("\n" + "=" * 70)
        print("🔄 APPLYING DATA AUGMENTATION (Training Set Only)")
        print("=" * 70)
        
        augmenter = DataAugmenter(**self.augmentation_params)
        
        # Current counts
        current_counts = Counter(y_train)
        class_names = ['Healthy', 'Stage1', 'Stage2', 'Stage3']
        
        # Target counts from paper (Table 1)
        target_counts = {
            0: 1258,  # Healthy
            1: 1219,  # Stage1
            2: 1264,  # Stage2
            3: 1197   # Stage3
        }
        
        print("\nCurrent vs Target counts:")
        for idx, name in enumerate(class_names):
            current = current_counts.get(idx, 0)
            target = target_counts.get(idx, 0)
            needed = target - current
            print(f"   {name}: {current} → {target} (need {needed} more)")
        
        augmented_images = list(X_train)
        augmented_labels = list(y_train)
        
        # Augment each class to reach target
        for class_idx in range(4):
            current = current_counts.get(class_idx, 0)
            target = target_counts.get(class_idx, 0)
            needed = target - current
            
            if needed <= 0:
                continue
            
            # Get images from this class
            class_images = X_train[y_train == class_idx]
            
            print(f"\n   Augmenting {class_names[class_idx]}...")
            
            # Generate augmented images
            aug_count = 0
            while aug_count < needed:
                # Randomly select an image
                idx = np.random.randint(0, len(class_images))
                original_img = class_images[idx]
                
                # Apply augmentation
                aug_img = augmenter.augment(original_img)
                
                augmented_images.append(aug_img)
                augmented_labels.append(class_idx)
                aug_count += 1
                
                if (aug_count) % 100 == 0:
                    print(f"      Generated {aug_count}/{needed}...")
        
        X_train_aug = np.array(augmented_images)
        y_train_aug = np.array(augmented_labels)
        
        # Verify final counts
        final_counts = Counter(y_train_aug)
        print("\n✅ Final Training Set Distribution:")
        for idx, name in enumerate(class_names):
            count = final_counts.get(idx, 0)
            target = target_counts.get(idx, 0)
            match = "✓" if count == target else f"(target: {target})"
            print(f"   {name}: {count} {match}")
        
        print(f"\nTotal training images: {len(X_train_aug)}")
        
        return X_train_aug, y_train_aug
    
    def save_dataset(self, X_train, X_test, y_train, y_test):
        """
        Save the processed dataset
        """
        print("\n" + "=" * 70)
        print("💾 SAVING PROCESSED DATASET")
        print("=" * 70)
        
        dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'class_names': ['Healthy', 
                           'Low Infestation (Stage 1)', 
                           'Medium Infestation (Stage 2)', 
                           'High Infestation (Stage 3)'],
            'n_classes': 4,
            'target_size': self.target_size,
            'methodology': 'exact_paper_implementation'
        }
        
        output_path = self.processed_data_path / 'dataset_paper_exact.pkl'
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n✅ Dataset saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return dataset
    
    def prepare(self):
        """
        Complete preparation pipeline - EXACT PAPER METHODOLOGY
        """
        print("\n" + "=" * 70)
        print("🌴 DATE PALM DATASET PREPARATION")
        print("   Following EXACT Paper Methodology")
        print("   Paper: Hessane et al. 2023 - BDMA")
        print("=" * 70)
        
        # Step 1: Load raw images (resized to 224x224)
        images, labels, filenames = self.load_raw_images()
        
        # Step 2: Preprocess (normalize)
        images_preprocessed = self.preprocess_images(images)
        
        # Step 3: Split into train/test (80/20)
        X_train, X_test, y_train, y_test = self.split_train_test(
            images_preprocessed, labels
        )
        
        # Step 4: Augment training data only
        X_train_aug, y_train_aug = self.augment_training_data(X_train, y_train)
        
        # Step 5: Save processed dataset
        dataset = self.save_dataset(X_train_aug, X_test, y_train_aug, y_test)
        
        print("\n" + "=" * 70)
        print("🎉 DATASET PREPARATION COMPLETE")
        print("=" * 70)
        print("\n📊 Final Dataset Summary:")
        print(f"   Training samples: {len(X_train_aug)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Image size: {self.target_size}")
        print(f"   Classes: 4")
        print(f"   Features to extract: 89 (80 GLCM + 9 HSV)")
        
        return dataset


def main():
    """
    Main execution
    """
    # Configuration
    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    
    # Check if raw data exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: Raw data path not found: {RAW_DATA_PATH}")
        print("\n📁 Please organize your dataset as follows:")
        print("data/raw/")
        print("  ├── Healthy/       (healthy leaflet images)")
        print("  ├── Stage1/        (low infestation)")
        print("  ├── Stage2/        (medium infestation)")
        print("  └── Stage3/        (high infestation)")
        return
    
    # Prepare dataset
    preparer = PaperExactDatasetPreparation(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    dataset = preparer.prepare()
    
    print("\n🚀 Next step: Run feature extraction and training")
    print("   python train_models_paper_exact.py")


if __name__ == "__main__":
    main()