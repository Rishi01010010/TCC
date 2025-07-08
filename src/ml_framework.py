"""
Machine Learning Framework for TCC Detection
Supporting AI/ML based algorithms as specified in project brief
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import pickle
from pathlib import Path
import json

# Optional ML framework imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .config import REQUIRED_FEATURES, TCC_IRBT_THRESHOLD


class TCCMLFramework:
    """
    Machine Learning framework for TCC detection and classification
    Supports multiple ML backends as specified in project brief
    """
    
    def __init__(self, framework: str = 'sklearn', model_directory: str = 'models'):
        """
        Initialize ML framework
        
        Args:
            framework: ML framework to use ('sklearn', 'tensorflow', 'pytorch')
            model_directory: Directory to save/load models
        """
        self.framework = framework.lower()
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Validate framework availability
        if self.framework == 'tensorflow' and not TF_AVAILABLE:
            print("Warning: TensorFlow not available, falling back to sklearn")
            self.framework = 'sklearn'
        elif self.framework == 'pytorch' and not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, falling back to sklearn")
            self.framework = 'sklearn'
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model based on selected framework"""
        if self.framework == 'sklearn':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.framework == 'tensorflow':
            self.model = self._create_tensorflow_model()
        elif self.framework == 'pytorch':
            self.model = self._create_pytorch_model()
    
    def _create_tensorflow_model(self):
        """Create TensorFlow CNN model for TCC detection"""
        if not TF_AVAILABLE:
            return None
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _create_pytorch_model(self):
        """Create PyTorch CNN model for TCC detection"""
        if not TORCH_AVAILABLE:
            return None
        
        class TCCNet(nn.Module):
            def __init__(self):
                super(TCCNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc1 = nn.Linear(128, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 1)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.relu(self.conv3(x))
                x = self.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x
        
        return TCCNet()
    
    def prepare_training_data(self, tcc_data: List[Dict], 
                            irbt_data_list: List[np.ndarray],
                            labels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML model
        
        Args:
            tcc_data: List of TCC detection results
            irbt_data_list: List of IRBT data arrays
            labels: Optional binary labels (1 for TCC, 0 for non-TCC)
            
        Returns:
            Tuple of (features, labels)
        """
        if self.framework in ['tensorflow', 'pytorch']:
            return self._prepare_image_data(tcc_data, irbt_data_list, labels)
        else:
            return self._prepare_feature_data(tcc_data, labels)
    
    def _prepare_feature_data(self, tcc_data: List[Dict], 
                            labels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature-based data for sklearn models"""
        features = []
        
        for tcc in tcc_data:
            feature_vector = []
            for feature_name in REQUIRED_FEATURES:
                if feature_name in tcc:
                    feature_vector.append(tcc[feature_name])
                else:
                    feature_vector.append(0.0)  # Missing feature
            
            # Add derived features
            feature_vector.extend([
                tcc.get('area_km2', 0.0),
                tcc.get('circularity', 0.0),
                tcc.get('max_radius_km', 0.0) / tcc.get('mean_radius_km', 1.0) if tcc.get('mean_radius_km', 0) > 0 else 0,
                abs(tcc.get('std_tb', 0.0)),
                tcc.get('pixel_count', 0),
            ])
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Generate labels if not provided (based on TCC criteria)
        if labels is None:
            labels = []
            for tcc in tcc_data:
                is_valid_tcc = (
                    tcc.get('area_km2', 0) >= 34800 and  # Area criterion
                    tcc.get('max_radius_km', 0) >= 111 and  # Radius criterion
                    tcc.get('min_tb', 999) <= TCC_IRBT_THRESHOLD and  # Temperature criterion
                    tcc.get('circularity', 0) >= 0.6  # Circularity criterion
                )
                labels.append(1 if is_valid_tcc else 0)
        
        return features, np.array(labels)
    
    def _prepare_image_data(self, tcc_data: List[Dict], 
                          irbt_data_list: List[np.ndarray],
                          labels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare image data for CNN models"""
        images = []
        
        for i, tcc in enumerate(tcc_data):
            if i < len(irbt_data_list):
                # Extract patch around TCC center
                irbt_data = irbt_data_list[i]
                patch = self._extract_tcc_patch(irbt_data, tcc)
                images.append(patch)
        
        images = np.array(images)
        
        # Add channel dimension for CNN
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        # Generate labels if not provided
        if labels is None:
            labels = []
            for tcc in tcc_data:
                is_valid_tcc = (
                    tcc.get('area_km2', 0) >= 34800 and
                    tcc.get('max_radius_km', 0) >= 111 and
                    tcc.get('min_tb', 999) <= TCC_IRBT_THRESHOLD and
                    tcc.get('circularity', 0) >= 0.6
                )
                labels.append(1 if is_valid_tcc else 0)
        
        return images, np.array(labels)
    
    def _extract_tcc_patch(self, irbt_data: np.ndarray, tcc: Dict, 
                          patch_size: int = 64) -> np.ndarray:
        """Extract square patch around TCC center"""
        # This would need proper implementation with coordinate mapping
        # For now, return a resized version of the data
        from skimage.transform import resize
        
        # Simple implementation - would need proper coordinate mapping
        patch = resize(irbt_data, (patch_size, patch_size), preserve_range=True)
        
        # Normalize to 0-1 range
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        
        return patch.astype(np.float32)
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              validation_split: float = 0.2, epochs: int = 50) -> Dict:
        """
        Train the ML model
        
        Args:
            features: Training features
            labels: Training labels
            validation_split: Fraction of data for validation
            epochs: Training epochs (for deep learning models)
            
        Returns:
            Training results dictionary
        """
        print(f"Training {self.framework} model...")
        print(f"Data shape: {features.shape}, Labels: {len(labels)}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        if self.framework == 'sklearn':
            return self._train_sklearn(X_train, X_val, y_train, y_val)
        elif self.framework == 'tensorflow':
            return self._train_tensorflow(X_train, X_val, y_train, y_val, epochs)
        elif self.framework == 'pytorch':
            return self._train_pytorch(X_train, X_val, y_train, y_val, epochs)
    
    def _train_sklearn(self, X_train, X_val, y_train, y_val) -> Dict:
        """Train sklearn model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_val_scaled)
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'classification_report': classification_report(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'feature_importance': self.model.feature_importances_.tolist()
        }
    
    def _train_tensorflow(self, X_train, X_val, y_train, y_val, epochs) -> Dict:
        """Train TensorFlow model"""
        if not TF_AVAILABLE:
            return {}
        
        # Normalize data
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        self.is_trained = True
        
        return {
            'history': {
                'loss': history.history['loss'],
                'accuracy': history.history['accuracy'],
                'val_loss': history.history['val_loss'],
                'val_accuracy': history.history['val_accuracy']
            },
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }
    
    def _train_pytorch(self, X_train, X_val, y_train, y_val, epochs) -> Dict:
        """Train PyTorch model"""
        if not TORCH_AVAILABLE:
            return {}
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            train_outputs = self.model(X_train)
            train_loss = criterion(train_outputs, y_train)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
                # Calculate accuracies
                train_pred = (train_outputs > 0.5).float()
                val_pred = (val_outputs > 0.5).float()
                train_acc = (train_pred == y_train).float().mean()
                val_acc = (val_pred == y_val).float().mean()
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                train_accuracies.append(train_acc.item())
                val_accuracies.append(val_acc.item())
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        self.is_trained = True
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'final_train_accuracy': train_accuracies[-1],
            'final_val_accuracy': val_accuracies[-1]
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            features: Input features
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.framework == 'sklearn':
            features_scaled = self.scaler.transform(features)
            return self.model.predict_proba(features_scaled)[:, 1]  # Probability of TCC
        
        elif self.framework == 'tensorflow':
            return self.model.predict(features).flatten()
        
        elif self.framework == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                predictions = self.model(features_tensor)
                return predictions.numpy().flatten()
    
    def save_model(self, model_name: str = 'tcc_model'):
        """Save trained model to disk"""
        model_path = self.model_directory / f"{model_name}_{self.framework}"
        
        if self.framework == 'sklearn':
            # Save sklearn model and scaler
            with open(f"{model_path}_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            with open(f"{model_path}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
        
        elif self.framework == 'tensorflow':
            self.model.save(f"{model_path}_model.h5")
        
        elif self.framework == 'pytorch':
            torch.save(self.model.state_dict(), f"{model_path}_model.pth")
        
        # Save metadata
        metadata = {
            'framework': self.framework,
            'is_trained': self.is_trained,
            'features': REQUIRED_FEATURES
        }
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str = 'tcc_model'):
        """Load trained model from disk"""
        model_path = self.model_directory / f"{model_name}_{self.framework}"
        
        try:
            if self.framework == 'sklearn':
                with open(f"{model_path}_model.pkl", 'rb') as f:
                    self.model = pickle.load(f)
                with open(f"{model_path}_scaler.pkl", 'rb') as f:
                    self.scaler = pickle.load(f)
            
            elif self.framework == 'tensorflow':
                self.model = tf.keras.models.load_model(f"{model_path}_model.h5")
            
            elif self.framework == 'pytorch':
                self.model.load_state_dict(torch.load(f"{model_path}_model.pth"))
            
            self.is_trained = True
            print(f"Model loaded from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def evaluate_tcc_detection(self, features: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        Evaluate model performance on TCC detection task
        
        Args:
            features: Test features
            true_labels: Ground truth labels
            
        Returns:
            Evaluation metrics dictionary
        """
        predictions = self.predict(features)
        binary_predictions = (predictions > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(true_labels, binary_predictions),
            'precision': precision_score(true_labels, binary_predictions),
            'recall': recall_score(true_labels, binary_predictions),
            'f1_score': f1_score(true_labels, binary_predictions),
            'classification_report': classification_report(true_labels, binary_predictions)
        } 