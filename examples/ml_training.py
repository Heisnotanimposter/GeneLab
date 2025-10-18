"""Machine learning training examples for GeneLab."""

import numpy as np
from sklearn.model_selection import train_test_split
from genelab import TreatmentPredictor
from genelab.utils import setup_logging, generate_training_data, vectorize_features

# Set up logging
setup_logging(level="INFO")

def generate_synthetic_data(n_samples=1000, n_features=100):
    """Generate synthetic data for demonstration."""
    # Generate random binary features
    X = np.random.randint(0, 2, size=(n_samples, n_features)).astype(float)
    
    # Generate labels (simple rule: more 1's = higher probability)
    y = (X.sum(axis=1) > n_features / 2).astype(int)
    
    # Add some noise
    noise = np.random.rand(n_samples) < 0.1
    y[noise] = 1 - y[noise]
    
    return X, y

def example_simple_classification():
    """Example: Train a simple classification model."""
    print("\n=== Example 1: Simple Classification ===")
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000, n_features=100)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Create and train model
    predictor = TreatmentPredictor(
        input_shape=(X_train.shape[1],),
        hidden_units=[64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    print("\nModel Architecture:")
    print(predictor.get_model_summary())
    
    # Train model
    history = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating on test set...")
    metrics = predictor.evaluate(X_test, y_test)
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Make predictions
    predictions = predictor.predict(X_test[:10])
    print(f"\nSample predictions (first 10):")
    for i, pred in enumerate(predictions[:10]):
        print(f"  Sample {i}: {pred[0]:.4f}")

def example_sequence_classification():
    """Example: Classify DNA sequences using k-mers."""
    print("\n=== Example 2: Sequence Classification ===")
    
    # Generate synthetic DNA sequences
    bases = ['A', 'T', 'C', 'G']
    sequences = []
    labels = []
    
    for i in range(500):
        # Generate random sequence
        seq = ''.join(np.random.choice(bases, size=100))
        sequences.append(seq)
        
        # Label based on GC content
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        labels.append(1 if gc_content > 0.5 else 0)
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Average length: {np.mean([len(s) for s in sequences]):.0f}")
    
    # Generate k-mers and vectorize
    kmers = generate_training_data(sequences, k=6)
    X, vectorizer = vectorize_features(kmers, ngram_range=(4, 4))
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Train model
    predictor = TreatmentPredictor(
        input_shape=(X_train.shape[1],),
        hidden_units=[128, 64],
        dropout_rate=0.4
    )
    
    history = predictor.train(
        X_train.toarray(), y_train,
        X_val.toarray(), y_val,
        epochs=30,
        batch_size=32
    )
    
    # Evaluate
    metrics = predictor.evaluate(X_test.toarray(), y_test)
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Plot training history
    print("\nPlotting training history...")
    predictor.plot_training_history(save_path="output/training_history.png")

def example_model_save_load():
    """Example: Save and load a trained model."""
    print("\n=== Example 3: Model Save/Load ===")
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=500, n_features=50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    predictor = TreatmentPredictor(input_shape=(X_train.shape[1],))
    predictor.train(X_train, y_train, X_test, y_test, epochs=10)
    
    # Save model
    model_path = "models/test_predictor"
    predictor.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Load model
    new_predictor = TreatmentPredictor(input_shape=(X_train.shape[1],))
    new_predictor.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Verify predictions are the same
    pred1 = predictor.predict(X_test[:5])
    pred2 = new_predictor.predict(X_test[:5])
    
    print(f"Predictions match: {np.allclose(pred1, pred2)}")

if __name__ == "__main__":
    import os
    
    # Create output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("GeneLab Machine Learning Examples")
    print("=" * 50)
    
    example_simple_classification()
    example_sequence_classification()
    example_model_save_load()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
