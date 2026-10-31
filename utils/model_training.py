"""
Model Training Module
Implements ML model training and evaluation
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Machine Learning model trainer
    """
    
    def __init__(self, model_type: str = 'svm', 
                 config: Optional[Dict] = None,
                 use_scaling: bool = True):
        """
        Initialize model trainer
        
        Args:
            model_type: Type of model ('svm', 'knn', 'rf', 'lightgbm')
            config: Model configuration dictionary
            use_scaling: Whether to use feature scaling
        """
        self.model_type = model_type.lower()
        self.config = config or {}
        self.use_scaling = use_scaling
        
        self.model = self._create_model()
        self.scaler = StandardScaler() if use_scaling else None
        self.is_fitted = False
    
    def _create_model(self):
        """
        Create model based on type
        
        Returns:
            Initialized model
        """
        if self.model_type == 'svm':
            return SVC(
                kernel=self.config.get('kernel', 'rbf'),
                C=self.config.get('C', 10),
                gamma=self.config.get('gamma', 'scale'),
                random_state=42,
                probability=True
            )
        
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=self.config.get('n_neighbors', 5),
                weights=self.config.get('weights', 'distance')
            )
        
        elif self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 20),
                min_samples_split=self.config.get('min_samples_split', 5),
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'lightgbm':
            return LGBMClassifier(
                num_leaves=self.config.get('num_leaves', 31),
                learning_rate=self.config.get('learning_rate', 0.05),
                n_estimators=self.config.get('n_estimators', 100),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics dictionary
        """
        # Scale features if needed
        if self.use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train_scaled)
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        # Calculate validation metrics if validation data provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val_scaled)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
        
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
        
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For models without predict_proba, return one-hot encoded predictions
            predictions = self.model.predict(X_scaled)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(predictions), n_classes))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Test metrics dictionary
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        metrics['probabilities'] = y_proba
        metrics['predictions'] = y_pred
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      n_splits: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            n_splits: Number of CV folds
            
        Returns:
            Cross-validation metrics
        """
        if self.use_scaling:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Calculate scores
        accuracy_scores = cross_val_score(self.model, X_scaled, y, 
                                         cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(self.model, X_scaled, y, 
                                          cv=cv, scoring='precision_macro')
        recall_scores = cross_val_score(self.model, X_scaled, y, 
                                       cv=cv, scoring='recall_macro')
        f1_scores = cross_val_score(self.model, X_scaled, y, 
                                   cv=cv, scoring='f1_macro')
        
        return {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'all_scores': {
                'accuracy': accuracy_scores,
                'precision': precision_scores,
                'recall': recall_scores,
                'f1': f1_scores
            }
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Metrics dictionary
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, 
                                                          output_dict=True, 
                                                          zero_division=0)
        }
    
    def save_model(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'config': self.config,
            'use_scaling': self.use_scaling
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded ModelTrainer instance
        """
        model_data = joblib.load(filepath)
        
        trainer = cls(
            model_type=model_data['model_type'],
            config=model_data['config'],
            use_scaling=model_data['use_scaling']
        )
        
        trainer.model = model_data['model']
        trainer.scaler = model_data['scaler']
        trainer.is_fitted = True
        
        return trainer


class MultiModelTrainer:
    """
    Train and compare multiple models
    """
    
    def __init__(self, model_configs: Dict[str, Dict]):
        """
        Initialize multi-model trainer
        
        Args:
            model_configs: Dictionary mapping model names to configs
        """
        self.model_configs = model_configs
        self.trainers = {}
        
        for name, config in model_configs.items():
            self.trainers[name] = ModelTrainer(
                model_type=name,
                config=config,
                use_scaling=True
            )
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of results for each model
        """
        results = {}
        
        for name, trainer in self.trainers.items():
            print(f"Training {name.upper()} model...")
            results[name] = trainer.train(X_train, y_train, X_val, y_val)
        
        return results
    
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate all models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of results for each model
        """
        results = {}
        
        for name, trainer in self.trainers.items():
            results[name] = trainer.evaluate(X_test, y_test)
        
        return results
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, ModelTrainer]:
        """
        Get best performing model
        
        Args:
            metric: Metric to compare ('accuracy', 'f1_score', etc.)
            
        Returns:
            Tuple of (model_name, model_trainer)
        """
        # This would need test results to determine best model
        # For now, return the first model
        name = list(self.trainers.keys())[0]
        return name, self.trainers[name]