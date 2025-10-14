"""LightGBM-based incident classification model."""

import pickle
import json
import os
import sys
from typing import Tuple, Dict, Any
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.synthetic_data import SyntheticIncidentGenerator
from src.features import IncidentFeatureExtractor


class IncidentClassifier:
    """Fast LightGBM-based incident classifier for 6 incident types."""

    INCIDENT_CLASSES = [
        'database_performance',
        'memory_leak',
        'network_latency',
        'disk_io',
        'cpu_spike',
        'unknown'
    ]

    CONFIDENCE_THRESHOLD = 0.75

    def __init__(self):
        """Initialize the incident classifier."""
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_extractor = IncidentFeatureExtractor()
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train LightGBM model with specified parameters.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)

        Returns:
            Training metrics and information
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)

        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # LightGBM parameters
        params = {
            'objective': 'multiclass',
            'num_class': len(self.INCIDENT_CLASSES),
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )

        self.is_trained = True

        # Calculate validation metrics
        val_predictions = self.model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_accuracy = np.mean(val_pred_classes == y_val)

        return {
            'validation_accuracy': val_accuracy,
            'num_features': X_train.shape[1],
            'num_classes': len(self.INCIDENT_CLASSES),
            'training_samples': len(X_train),
            'best_iteration': self.model.best_iteration
        }

    def predict(self, features: np.ndarray) -> Tuple[str, float, bool]:
        """Predict incident class with confidence and edge case detection.

        Args:
            features: Feature vector (15,) or batch (n_samples, 15)

        Returns:
            Tuple of (predicted_class, confidence, is_edge_case)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Handle single sample vs batch
        if features.ndim == 1:
            features = features.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        # Get predictions
        predictions = self.model.predict(features)

        if single_sample:
            # Single prediction
            probs = predictions[0]
            predicted_class_idx = np.argmax(probs)
            confidence = float(probs[predicted_class_idx])
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]

            # Determine if edge case
            is_edge_case = (confidence < self.CONFIDENCE_THRESHOLD) or (predicted_class == 'unknown')

            return predicted_class, confidence, is_edge_case
        else:
            # Batch predictions - return first sample for compatibility
            probs = predictions[0]
            predicted_class_idx = np.argmax(probs)
            confidence = float(probs[predicted_class_idx])
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            is_edge_case = (confidence < self.CONFIDENCE_THRESHOLD) or (predicted_class == 'unknown')

            return predicted_class, confidence, is_edge_case

    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict batch of incidents.

        Args:
            features: Feature matrix (n_samples, 15)

        Returns:
            Tuple of (predicted_classes, confidences, is_edge_cases)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Get predictions
        predictions = self.model.predict(features)

        # Extract results
        predicted_class_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        predicted_classes = self.label_encoder.inverse_transform(predicted_class_indices)

        # Determine edge cases
        is_edge_cases = (confidences < self.CONFIDENCE_THRESHOLD) | (predicted_classes == 'unknown')

        return predicted_classes, confidences, is_edge_cases

    def save(self, path: str) -> None:
        """Save model and metadata to disk.

        Args:
            path: Directory path to save model files
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        os.makedirs(path, exist_ok=True)

        # Save LightGBM model
        model_path = os.path.join(path, 'model.txt')
        self.model.save_model(model_path)

        # Save label encoder
        encoder_path = os.path.join(path, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Save metadata
        metadata = {
            'incident_classes': self.INCIDENT_CLASSES,
            'confidence_threshold': self.CONFIDENCE_THRESHOLD,
            'num_features': 15,
            'model_type': 'lightgbm_multiclass'
        }

        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Directory path containing model files
        """
        # Load LightGBM model
        model_path = os.path.join(path, 'model.txt')
        self.model = lgb.Booster(model_file=model_path)

        # Load label encoder
        encoder_path = os.path.join(path, 'label_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load and validate metadata
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Validate metadata
        assert metadata['incident_classes'] == self.INCIDENT_CLASSES
        assert metadata['confidence_threshold'] == self.CONFIDENCE_THRESHOLD

        self.is_trained = True


def _generate_training_labels(incidents) -> np.ndarray:
    """Generate training labels from ground truth data."""
    labels = []

    for incident in incidents:
        # Extract actual root cause from ground truth
        actual_cause = incident.ground_truth.get('actual_root_cause', 'unknown')

        # Map specific causes to general categories
        if 'database' in actual_cause or 'db_' in actual_cause:
            label = 'database_performance'
        elif 'memory' in actual_cause or 'leak' in actual_cause:
            label = 'memory_leak'
        elif 'network' in actual_cause or 'latency' in actual_cause:
            label = 'network_latency'
        elif 'disk' in actual_cause or 'io' in actual_cause:
            label = 'disk_io'
        elif 'cpu' in actual_cause or 'spike' in actual_cause:
            label = 'cpu_spike'
        else:
            label = 'unknown'

        labels.append(label)

    return np.array(labels)


if __name__ == "__main__":
    print("Training IncidentClassifier...")
    print("=" * 40)

    # Generate training data
    print("Generating 1000 training incidents...")
    generator = SyntheticIncidentGenerator(seed=42)
    incidents = generator.generate_training_dataset(n_samples=1000)

    # Extract features
    print("Extracting features...")
    extractor = IncidentFeatureExtractor()
    features = extractor.extract_batch_features(incidents)

    # Generate labels from ground truth
    print("Generating labels from ground truth...")
    labels = _generate_training_labels(incidents)

    print(f"Feature shape: {features.shape}")
    print(f"Label distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")

    # Train model
    print("\nTraining LightGBM model...")
    classifier = IncidentClassifier()
    training_metrics = classifier.train(features, labels)

    print(f"Training complete!")
    print(f"Validation accuracy: {training_metrics['validation_accuracy']:.3f}")
    print(f"Best iteration: {training_metrics['best_iteration']}")

    # Test prediction speed
    print("\nTesting prediction speed...")
    test_incident = incidents[0]
    test_features = extractor.extract_model_features(test_incident)

    import time
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        pred_class, confidence, is_edge = classifier.predict(test_features)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    print(f"Average prediction time: {avg_time:.3f}ms")
    print(f"Sample prediction: {pred_class} (confidence: {confidence:.3f}, edge_case: {is_edge})")

    if avg_time < 10.0:
        print(f"[PASS] Average time ({avg_time:.3f}ms) < 10ms target")
    else:
        print(f"[FAIL] Average time ({avg_time:.3f}ms) >= 10ms target")

    # Save model
    print("\nSaving model to ./models/incident_classifier...")
    classifier.save('./models/incident_classifier')
    print("Model saved successfully!")

    # Test loading
    print("\nTesting model loading...")
    new_classifier = IncidentClassifier()
    new_classifier.load('./models/incident_classifier')

    # Verify loaded model works
    pred_class_2, confidence_2, is_edge_2 = new_classifier.predict(test_features)
    print(f"Loaded model prediction: {pred_class_2} (confidence: {confidence_2:.3f}, edge_case: {is_edge_2})")

    print("\nTraining and testing complete!")