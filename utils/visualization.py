"""
Visualization Module
Implements plotting and visualization functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Visualization utilities for the application
    """
    
    def __init__(self, class_names: List[str], class_colors: Optional[List[str]] = None):
        """
        Initialize visualizer
        
        Args:
            class_names: List of class names
            class_colors: List of colors for each class
        """
        self.class_names = class_names
        self.n_classes = len(class_names)
        
        if class_colors is None:
            self.class_colors = px.colors.qualitative.Set2[:self.n_classes]
        else:
            self.class_colors = class_colors
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             title: str = "Confusion Matrix",
                             normalize: bool = False) -> go.Figure:
        """
        Plot confusion matrix using Plotly
        
        Args:
            cm: Confusion matrix
            title: Plot title
            normalize: Whether to normalize the matrix
            
        Returns:
            Plotly figure
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            text_format = '.2%'
        else:
            text_format = 'd'
        
        # Create annotations
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                if normalize:
                    text = f"{cm[i][j]:.1%}"
                else:
                    text = str(int(cm[i][j]))
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=text,
                        showarrow=False,
                        font=dict(color='white' if cm[i][j] > cm.max()/2 else 'black',
                                 size=14)
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.class_names,
            y=self.class_names,
            colorscale='Blues',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis=dict(title='Predicted Label', side='bottom'),
            yaxis=dict(title='True Label', autorange='reversed'),
            width=600,
            height=600,
            annotations=annotations,
            font=dict(size=12)
        )
        
        return fig
    
    def plot_classification_report(self, report: Dict) -> go.Figure:
        """
        Plot classification report as grouped bar chart
        
        Args:
            report: Classification report dictionary
            
        Returns:
            Plotly figure
        """
        # Extract metrics for each class
        classes = [str(i) for i in range(self.n_classes)]
        metrics = ['precision', 'recall', 'f1-score']
        
        data = []
        for metric in metrics:
            values = [report[cls][metric] for cls in classes]
            data.append(go.Bar(
                name=metric.capitalize(),
                x=self.class_names,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        fig = go.Figure(data=data)
        
        fig.update_layout(
            title=dict(text='Classification Report by Class', x=0.5, xanchor='center'),
            xaxis=dict(title='Class'),
            yaxis=dict(title='Score', range=[0, 1.1]),
            barmode='group',
            width=800,
            height=500,
            font=dict(size=12),
            legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
        )
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict],
                             metric: str = 'accuracy') -> go.Figure:
        """
        Compare multiple models on a specific metric
        
        Args:
            results: Dictionary of model results
            metric: Metric to compare
            
        Returns:
            Plotly figure
        """
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        
        # Create color scale based on performance
        colors = ['rgb(158,202,225)' if v < max(values) else 'rgb(8,81,156)' 
                  for v in values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=values,
                text=[f'{v:.2%}' for v in values],
                textposition='auto',
                marker=dict(color=colors),
                hovertemplate='<b>%{x}</b><br>' + 
                             metric.capitalize() + ': %{y:.2%}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text=f'Model Comparison - {metric.capitalize()}', 
                      x=0.5, xanchor='center'),
            xaxis=dict(title='Model'),
            yaxis=dict(title=metric.capitalize(), range=[0, 1.1]),
            width=700,
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str],
                               importances: np.ndarray,
                               top_n: int = 20) -> go.Figure:
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            top_n: Number of top features to show
            
        Returns:
            Plotly figure
        """
        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig = go.Figure(data=[
            go.Bar(
                y=top_features[::-1],  # Reverse for better display
                x=top_importances[::-1],
                orientation='h',
                marker=dict(color=top_importances[::-1],
                          colorscale='Viridis',
                          showscale=True),
                text=[f'{v:.4f}' for v in top_importances[::-1]],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=dict(text=f'Top {top_n} Most Important Features',
                      x=0.5, xanchor='center'),
            xaxis=dict(title='Importance'),
            yaxis=dict(title='Feature'),
            width=800,
            height=max(500, top_n * 25),
            font=dict(size=10)
        )
        
        return fig
    
    def plot_training_history(self, history: Dict) -> go.Figure:
        """
        Plot training history (loss/accuracy over epochs)
        
        Args:
            history: Training history dictionary
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy', 'Model Loss')
        )
        
        # Accuracy plot
        if 'accuracy' in history:
            fig.add_trace(
                go.Scatter(x=list(range(len(history['accuracy']))),
                          y=history['accuracy'],
                          name='Train Accuracy',
                          mode='lines+markers'),
                row=1, col=1
            )
        
        if 'val_accuracy' in history:
            fig.add_trace(
                go.Scatter(x=list(range(len(history['val_accuracy']))),
                          y=history['val_accuracy'],
                          name='Val Accuracy',
                          mode='lines+markers'),
                row=1, col=1
            )
        
        # Loss plot
        if 'loss' in history:
            fig.add_trace(
                go.Scatter(x=list(range(len(history['loss']))),
                          y=history['loss'],
                          name='Train Loss',
                          mode='lines+markers'),
                row=1, col=2
            )
        
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(x=list(range(len(history['val_loss']))),
                          y=history['val_loss'],
                          name='Val Loss',
                          mode='lines+markers'),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        fig.update_layout(
            title=dict(text='Training History', x=0.5, xanchor='center'),
            width=1000,
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> go.Figure:
        """
        Plot ROC curves for multiclass classification
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Plotly figure
        """
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        fig = go.Figure()
        
        # Plot ROC curve for each class
        for i in range(self.n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{self.class_names[i]} (AUC = {roc_auc:.3f})',
                mode='lines',
                line=dict(width=2)
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(dash='dash', color='gray', width=2)
        ))
        
        fig.update_layout(
            title=dict(text='ROC Curves - Multiclass Classification',
                      x=0.5, xanchor='center'),
            xaxis=dict(title='False Positive Rate', range=[0, 1]),
            yaxis=dict(title='True Positive Rate', range=[0, 1]),
            width=800,
            height=600,
            font=dict(size=12),
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig
    
    def plot_prediction_distribution(self, predictions: np.ndarray,
                                     true_labels: Optional[np.ndarray] = None) -> go.Figure:
        """
        Plot distribution of predictions
        
        Args:
            predictions: Model predictions
            true_labels: True labels (optional)
            
        Returns:
            Plotly figure
        """
        # Count predictions for each class
        unique, counts = np.unique(predictions, return_counts=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=[self.class_names[i] for i in unique],
                y=counts,
                marker=dict(color=[self.class_colors[i] for i in unique]),
                text=counts,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text='Prediction Distribution', x=0.5, xanchor='center'),
            xaxis=dict(title='Class'),
            yaxis=dict(title='Number of Samples'),
            width=700,
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def visualize_image_with_prediction(self, image: np.ndarray,
                                       prediction: int,
                                       probabilities: np.ndarray,
                                       true_label: Optional[int] = None) -> go.Figure:
        """
        Visualize image with prediction and probabilities
        
        Args:
            image: Input image
            prediction: Predicted class
            probabilities: Class probabilities
            true_label: True label (optional)
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.5, 0.5],
            subplot_titles=('Input Image', 'Prediction Probabilities'),
            specs=[[{"type": "image"}, {"type": "bar"}]]
        )
        
        # Prepare image for display
        if image.max() <= 1.0:
            display_image = (image * 255).astype(np.uint8)
        else:
            display_image = image.astype(np.uint8)
        
        # Add image
        fig.add_trace(
            go.Image(z=display_image),
            row=1, col=1
        )
        
        # Add probability bars
        colors = ['green' if i == prediction else 'lightblue' 
                 for i in range(self.n_classes)]
        
        fig.add_trace(
            go.Bar(
                x=self.class_names,
                y=probabilities,
                marker=dict(color=colors),
                text=[f'{p:.1%}' for p in probabilities],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Update layout
        title_text = f"Predicted: {self.class_names[prediction]}"
        if true_label is not None:
            title_text += f" | True: {self.class_names[true_label]}"
        
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor='center'),
            width=1000,
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=2)
        
        return fig
    
    def plot_glcm_features(self, feature_dict: Dict) -> go.Figure:
        """
        Plot GLCM features as heatmap
        
        Args:
            feature_dict: Dictionary of GLCM features grouped by property
            
        Returns:
            Plotly figure
        """
        properties = list(feature_dict.keys())
        n_props = len(properties)
        
        fig = make_subplots(
            rows=1, cols=n_props,
            subplot_titles=properties,
            horizontal_spacing=0.1
        )
        
        for idx, prop in enumerate(properties, 1):
            values = np.array(feature_dict[prop]).reshape(4, 4)  # 4 distances x 4 angles
            
            fig.add_trace(
                go.Heatmap(
                    z=values,
                    colorscale='Viridis',
                    showscale=(idx == n_props),
                    x=['0°', '45°', '90°', '135°'],
                    y=['d=1', 'd=3', 'd=6', 'd=9']
                ),
                row=1, col=idx
            )
        
        fig.update_layout(
            title=dict(text='GLCM Feature Visualization', x=0.5, xanchor='center'),
            width=300 * n_props,
            height=400,
            font=dict(size=10)
        )
        
        return fig
    
    def plot_hsv_features(self, feature_dict: Dict) -> go.Figure:
        """
        Plot HSV color moment features
        
        Args:
            feature_dict: Dictionary of HSV features
            
        Returns:
            Plotly figure
        """
        channels = ['H', 'S', 'V']
        moments = ['mean', 'std', 'skewness']
        
        data = []
        for channel in channels:
            values = [feature_dict[f'HSV_{channel}_{moment}'] 
                     for moment in moments]
            data.append(go.Bar(
                name=channel,
                x=moments,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig = go.Figure(data=data)
        
        fig.update_layout(
            title=dict(text='HSV Color Moment Features', x=0.5, xanchor='center'),
            xaxis=dict(title='Moment Type'),
            yaxis=dict(title='Value'),
            barmode='group',
            width=700,
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def plot_dataset_distribution(self, labels: np.ndarray) -> go.Figure:
        """
        Plot dataset class distribution
        
        Args:
            labels: Array of labels
            
        Returns:
            Plotly figure
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        fig = go.Figure(data=[
            go.Pie(
                labels=[self.class_names[i] for i in unique],
                values=counts,
                marker=dict(colors=[self.class_colors[i] for i in unique]),
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text='Dataset Class Distribution', x=0.5, xanchor='center'),
            width=600,
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def create_metrics_table(self, metrics: Dict) -> pd.DataFrame:
        """
        Create formatted metrics table
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Pandas DataFrame
        """
        data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}"
            ]
        }
        
        return pd.DataFrame(data)


def display_image_grid(images: List[np.ndarray], 
                      labels: Optional[List[int]] = None,
                      predictions: Optional[List[int]] = None,
                      class_names: Optional[List[str]] = None,
                      n_cols: int = 4) -> plt.Figure:
    """
    Display images in a grid
    
    Args:
        images: List of images
        labels: True labels (optional)
        predictions: Predicted labels (optional)
        class_names: List of class names (optional)
        n_cols: Number of columns
        
    Returns:
        Matplotlib figure
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < n_images:
            img = images[idx]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            ax.imshow(img)
            ax.axis('off')
            
            # Add title with label/prediction
            title = ""
            if labels is not None and class_names is not None:
                title += f"True: {class_names[labels[idx]]}"
            if predictions is not None and class_names is not None:
                if title:
                    title += "\n"
                title += f"Pred: {class_names[predictions[idx]]}"
                
                # Color based on correctness
                if labels is not None:
                    color = 'green' if predictions[idx] == labels[idx] else 'red'
                    ax.set_title(title, color=color, fontsize=10, weight='bold')
                else:
                    ax.set_title(title, fontsize=10)
            elif title:
                ax.set_title(title, fontsize=10)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    return fig