"""
Feature Extraction Module
Implements GLCM and HSV feature extraction
FIXED: Proper data type handling for OpenCV
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class GLCMFeatureExtractor:
    """
    Extract GLCM (Gray Level Co-occurrence Matrix) texture features
    """
    
    def __init__(self, distances: List[int] = [1, 3, 6, 9], 
                 angles: List[float] = [0, 45, 90, 135],
                 properties: List[str] = ['contrast', 'dissimilarity', 
                                          'homogeneity', 'energy', 'correlation']):
        """
        Initialize GLCM feature extractor
        
        Args:
            distances: List of pixel pair distances
            angles: List of angles in degrees
            properties: List of GLCM properties to compute
        """
        self.distances = distances
        self.angles = [np.radians(angle) for angle in angles]
        self.properties = properties
        self.n_features = len(distances) * len(angles) * len(properties)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract GLCM features from image
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Feature vector of shape (80,)
        """
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            # Ensure uint8 before color conversion
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Ensure grayscale is also uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Compute GLCM matrix
        glcm = graycomatrix(
            image, 
            distances=self.distances,
            angles=self.angles,
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Extract features
        features = []
        for prop in self.properties:
            feature = graycoprops(glcm, prop)
            features.append(feature.flatten())
        
        # Concatenate all features
        feature_vector = np.concatenate(features)
        
        return feature_vector
    
    def get_feature_names(self) -> List[str]:
        """Get names of all GLCM features"""
        feature_names = []
        for prop in self.properties:
            for dist in self.distances:
                for angle in [0, 45, 90, 135]:
                    feature_names.append(f"GLCM_{prop}_d{dist}_a{angle}")
        return feature_names


class HSVColorMomentExtractor:
    """
    Extract HSV color moment features
    """
    
    def __init__(self):
        """Initialize HSV color moment extractor"""
        self.channels = ['H', 'S', 'V']
        self.moments = ['mean', 'std', 'skewness']
        self.n_features = 9
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HSV color moment features
        
        Args:
            image: Input RGB image
            
        Returns:
            Feature vector of shape (9,)
        """
        # Ensure uint8 before HSV conversion
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        features = []
        
        # Extract moments for each channel
        for channel_idx in range(3):
            channel = hsv_image[:, :, channel_idx].flatten().astype(np.float64)
            
            # Mean
            mean_val = np.mean(channel)
            features.append(mean_val)
            
            # Standard Deviation
            std_val = np.std(channel)
            features.append(std_val)
            
            # Skewness
            skewness_val = skew(channel)
            features.append(skewness_val)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all HSV features"""
        feature_names = []
        for channel in self.channels:
            for moment in self.moments:
                feature_names.append(f"HSV_{channel}_{moment}")
        return feature_names


class CombinedFeatureExtractor:
    """
    Combined GLCM and HSV feature extractor
    """
    
    def __init__(self, use_glcm: bool = True, use_hsv: bool = True):
        """
        Initialize combined feature extractor
        
        Args:
            use_glcm: Whether to use GLCM features
            use_hsv: Whether to use HSV features
        """
        self.use_glcm = use_glcm
        self.use_hsv = use_hsv
        
        self.glcm_extractor = GLCMFeatureExtractor() if use_glcm else None
        self.hsv_extractor = HSVColorMomentExtractor() if use_hsv else None
        
        self.n_features = 0
        if use_glcm:
            self.n_features += self.glcm_extractor.n_features
        if use_hsv:
            self.n_features += self.hsv_extractor.n_features
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract combined features from image
        
        Args:
            image: Input RGB image (can be float or uint8)
            
        Returns:
            Combined feature vector
        """
        features = []
        
        if self.use_glcm:
            glcm_features = self.glcm_extractor.extract_features(image)
            features.append(glcm_features)
        
        if self.use_hsv:
            hsv_features = self.hsv_extractor.extract_features(image)
            features.append(hsv_features)
        
        return np.concatenate(features) if len(features) > 1 else features[0]
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        feature_names = []
        
        if self.use_glcm:
            feature_names.extend(self.glcm_extractor.get_feature_names())
        
        if self.use_hsv:
            feature_names.extend(self.hsv_extractor.get_feature_names())
        
        return feature_names
    
    def extract_batch(self, images: List[np.ndarray], 
                     show_progress: bool = False) -> np.ndarray:
        """
        Extract features from multiple images
        
        Args:
            images: List of images
            show_progress: Whether to show progress bar
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        features_list = []
        
        if show_progress:
            try:
                import streamlit as st
                progress_bar = st.progress(0)
                for idx, img in enumerate(images):
                    features = self.extract_features(img)
                    features_list.append(features)
                    progress_bar.progress((idx + 1) / len(images))
            except:
                for img in images:
                    features = self.extract_features(img)
                    features_list.append(features)
        else:
            for img in images:
                features = self.extract_features(img)
                features_list.append(features)
        
        return np.array(features_list)


def visualize_glcm_features(image: np.ndarray) -> Dict:
    """Visualize GLCM features for a single image"""
    extractor = GLCMFeatureExtractor()
    features = extractor.extract_features(image)
    feature_names = extractor.get_feature_names()
    
    feature_dict = {}
    for name, value in zip(feature_names, features):
        prop = name.split('_')[1]
        if prop not in feature_dict:
            feature_dict[prop] = []
        feature_dict[prop].append(value)
    
    return feature_dict


def visualize_hsv_features(image: np.ndarray) -> Dict:
    """Visualize HSV features for a single image"""
    extractor = HSVColorMomentExtractor()
    features = extractor.extract_features(image)
    feature_names = extractor.get_feature_names()
    
    return dict(zip(feature_names, features))