"""
Data Preprocessing Module
Handles image loading, augmentation, and preprocessing
"""

import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ImagePreprocessor:
    """
    Image preprocessing utilities
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        """
        Initialize image preprocessor
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.array(image)
        return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1]
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if image.max() > 1.0:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image (can be path or array)
            
        Returns:
            Preprocessed image
        """
        # Load if path
        if isinstance(image, (str, Path)):
            image = self.load_image(str(image))
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        if self.normalize:
            image = self.normalize_image(image)
        
        return image
    
    def preprocess_batch(self, images: List, show_progress: bool = False) -> np.ndarray:
        """
        Preprocess multiple images
        
        Args:
            images: List of images or paths
            show_progress: Whether to show progress
            
        Returns:
            Array of preprocessed images
        """
        processed = []
        
        if show_progress:
            try:
                import streamlit as st
                progress_bar = st.progress(0)
                for idx, img in enumerate(images):
                    processed.append(self.preprocess(img))
                    progress_bar.progress((idx + 1) / len(images))
            except:
                for img in images:
                    processed.append(self.preprocess(img))
        else:
            for img in images:
                processed.append(self.preprocess(img))
        
        return np.array(processed)


class DataAugmenter:
    """
    Data augmentation utilities
    """
    
    def __init__(self, rotation_range: int = 40,
                 width_shift_range: float = 0.2,
                 height_shift_range: float = 0.2,
                 zoom_range: float = 0.2,
                 horizontal_flip: bool = True):
        """
        Initialize data augmenter
        
        Args:
            rotation_range: Degree range for random rotations
            width_shift_range: Fraction of width for horizontal shifts
            height_shift_range: Fraction of height for vertical shifts
            zoom_range: Range for random zoom
            horizontal_flip: Whether to randomly flip images horizontally
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
    
    def rotate_image(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        Rotate image by random or specified angle
        
        Args:
            image: Input image
            angle: Rotation angle (random if None)
            
        Returns:
            Rotated image
        """
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                 borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """
        Flip image horizontally
        
        Args:
            image: Input image
            
        Returns:
            Flipped image
        """
        return cv2.flip(image, 1)
    
    def shift_image(self, image: np.ndarray, 
                   width_shift: Optional[float] = None,
                   height_shift: Optional[float] = None) -> np.ndarray:
        """
        Shift image horizontally and vertically
        
        Args:
            image: Input image
            width_shift: Horizontal shift fraction (random if None)
            height_shift: Vertical shift fraction (random if None)
            
        Returns:
            Shifted image
        """
        height, width = image.shape[:2]
        
        if width_shift is None:
            width_shift = np.random.uniform(-self.width_shift_range, 
                                           self.width_shift_range)
        if height_shift is None:
            height_shift = np.random.uniform(-self.height_shift_range, 
                                            self.height_shift_range)
        
        width_shift_pixels = int(width * width_shift)
        height_shift_pixels = int(height * height_shift)
        
        translation_matrix = np.float32([
            [1, 0, width_shift_pixels],
            [0, 1, height_shift_pixels]
        ])
        
        shifted = cv2.warpAffine(image, translation_matrix, (width, height),
                                borderMode=cv2.BORDER_REFLECT)
        
        return shifted
    
    def zoom_image(self, image: np.ndarray, 
                   zoom_factor: Optional[float] = None) -> np.ndarray:
        """
        Zoom into image
        
        Args:
            image: Input image
            zoom_factor: Zoom factor (random if None)
            
        Returns:
            Zoomed image
        """
        if zoom_factor is None:
            zoom_factor = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        new_height = int(height / zoom_factor)
        new_width = int(width / zoom_factor)
        
        # Calculate crop region
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        
        # Crop and resize
        cropped = image[top:top+new_height, left:left+new_width]
        zoomed = cv2.resize(cropped, (width, height))
        
        return zoomed
    
    def augment(self, image: np.ndarray, 
                apply_rotation: bool = True,
                apply_flip: bool = True,
                apply_shift: bool = True,
                apply_zoom: bool = True) -> np.ndarray:
        """
        Apply random augmentations to image
        
        Args:
            image: Input image
            apply_rotation: Whether to apply rotation
            apply_flip: Whether to apply horizontal flip
            apply_shift: Whether to apply shifts
            apply_zoom: Whether to apply zoom
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        if apply_rotation and np.random.random() > 0.5:
            augmented = self.rotate_image(augmented)
        
        if apply_flip and self.horizontal_flip and np.random.random() > 0.5:
            augmented = self.flip_horizontal(augmented)
        
        if apply_shift and np.random.random() > 0.5:
            augmented = self.shift_image(augmented)
        
        if apply_zoom and np.random.random() > 0.5:
            augmented = self.zoom_image(augmented)
        
        return augmented
    
    def augment_batch(self, images: List[np.ndarray], 
                     n_augmentations: int = 1) -> List[np.ndarray]:
        """
        Augment multiple images
        
        Args:
            images: List of input images
            n_augmentations: Number of augmented versions per image
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for image in images:
            augmented_images.append(image)  # Original
            
            for _ in range(n_augmentations):
                aug = self.augment(image)
                augmented_images.append(aug)
        
        return augmented_images


def load_dataset_from_folder(data_dir: str, 
                             class_names: List[str],
                             preprocessor: Optional[ImagePreprocessor] = None) -> Tuple:
    """
    Load dataset from folder structure
    
    Expected structure:
    data_dir/
        class_0/
            img1.jpg
            img2.jpg
        class_1/
            img1.jpg
            img2.jpg
    
    Args:
        data_dir: Root directory of dataset
        class_names: List of class names
        preprocessor: Image preprocessor instance
        
    Returns:
        Tuple of (images, labels)
    """
    if preprocessor is None:
        preprocessor = ImagePreprocessor()
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                image = preprocessor.preprocess(img_path)
                images.append(image)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)