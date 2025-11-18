"""
Utility functions for Edge AI Recyclable Classifier
Data loading, preprocessing, visualization, and evaluation helpers
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import config

# ============================================================================
# Data Loading
# ============================================================================

def create_data_generators(train_dir=None, val_dir=None, test_dir=None):
    """
    Create ImageDataGenerators for training, validation, and testing

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory (optional)

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    train_dir = train_dir or config.TRAIN_DIR
    val_dir = val_dir or config.VAL_DIR

    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(**config.AUGMENTATION_CONFIG)

    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(**config.VAL_AUGMENTATION_CONFIG)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_datagen = ImageDataGenerator(**config.VAL_AUGMENTATION_CONFIG)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    if test_generator:
        print(f"✓ Test samples: {test_generator.samples}")
    print(f"✓ Classes: {train_generator.class_indices}")

    return train_generator, val_generator, test_generator


def load_and_preprocess_image(image_path, target_size=None):
    """
    Load and preprocess a single image for inference

    Args:
        image_path: Path to image file or PIL Image
        target_size: Tuple of (height, width)

    Returns:
        Preprocessed image array (1, H, W, 3)
    """
    from PIL import Image

    target_size = target_size or config.IMG_SIZE

    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path  # Assume PIL Image

    # Resize
    img = img.resize(target_size)

    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves

    Args:
        history: Keras History object or dict
        save_path: Path to save the plot (optional)
    """
    if isinstance(history, dict):
        hist_dict = history
    else:
        hist_dict = history.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(hist_dict['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(hist_dict['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(hist_dict['loss'], label='Train Loss', marker='o')
    axes[1].plot(hist_dict['val_loss'], label='Val Loss', marker='s')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training plots saved to {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix

    Args:
        y_true: True labels (integers)
        y_pred: Predicted labels (integers)
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    class_names = class_names or config.CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")

    plt.show()

    return cm


def plot_sample_predictions(model, data_generator, num_samples=9, class_names=None):
    """
    Plot sample predictions from the model

    Args:
        model: Trained Keras model or TFLite interpreter
        data_generator: Keras ImageDataGenerator
        num_samples: Number of samples to display
        class_names: List of class names
    """
    class_names = class_names or config.CLASS_NAMES

    # Get a batch of images
    images, labels = next(data_generator)
    predictions = model.predict(images[:num_samples])

    # Plot
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        ax = axes[i]

        # Display image
        ax.imshow(images[i])

        # Get prediction
        true_label = class_names[np.argmax(labels[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i])

        # Color: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'

        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                     color=color, fontsize=10)
        ax.axis('off')

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, data_generator, class_names=None):
    """
    Evaluate model and print detailed metrics

    Args:
        model: Trained Keras model
        data_generator: Keras ImageDataGenerator
        class_names: List of class names

    Returns:
        dict: Evaluation metrics
    """
    class_names = class_names or config.CLASS_NAMES

    print("Evaluating model...")

    # Get predictions
    predictions = model.predict(data_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = data_generator.classes

    # Overall metrics
    loss, accuracy = model.evaluate(data_generator, verbose=0)

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"  {class_name}: {class_acc:.4f}")

    print("="*60)

    metrics = {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_true, y_pred,
                                                        target_names=class_names,
                                                        output_dict=True)
    }

    return metrics


# ============================================================================
# Model Utilities
# ============================================================================

def save_training_history(history, save_path=None):
    """
    Save training history to JSON file

    Args:
        history: Keras History object or dict
        save_path: Path to save JSON file
    """
    save_path = save_path or config.HISTORY_JSON_PATH

    if isinstance(history, dict):
        hist_dict = history
    else:
        hist_dict = history.history

    # Convert numpy types to Python types
    hist_dict = {k: [float(x) for x in v] for k, v in hist_dict.items()}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)

    print(f"✓ Training history saved to {save_path}")


def load_training_history(load_path=None):
    """
    Load training history from JSON file

    Args:
        load_path: Path to JSON file

    Returns:
        dict: Training history
    """
    load_path = load_path or config.HISTORY_JSON_PATH

    with open(load_path, 'r') as f:
        history = json.load(f)

    return history


def save_metrics(metrics, save_path=None):
    """
    Save performance metrics to JSON file

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON file
    """
    save_path = save_path or config.METRICS_JSON_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved to {save_path}")


def get_model_size(model_path):
    """
    Get model file size in MB

    Args:
        model_path: Path to model file

    Returns:
        float: Size in MB
    """
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    else:
        return 0.0


def print_model_summary(model):
    """
    Print model architecture summary

    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("Model Architecture")
    print("="*60)
    model.summary()
    print("="*60)

    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("="*60 + "\n")


# ============================================================================
# Representative Dataset for Quantization
# ============================================================================

def representative_dataset_generator(data_generator, num_samples=100):
    """
    Generator function for representative dataset (INT8 quantization)

    Args:
        data_generator: Keras ImageDataGenerator
        num_samples: Number of samples to use

    Yields:
        list: Single image batch for calibration
    """
    data_generator.reset()
    count = 0

    for images, _ in data_generator:
        for img in images:
            if count >= num_samples:
                return
            yield [np.expand_dims(img, axis=0).astype(np.float32)]
            count += 1

        if count >= num_samples:
            break


# ============================================================================
# Dataset Information
# ============================================================================

def analyze_dataset(data_dir):
    """
    Analyze dataset and print statistics

    Args:
        data_dir: Path to dataset directory
    """
    print("\n" + "="*60)
    print(f"Dataset Analysis: {data_dir}")
    print("="*60)

    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    class_dirs = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d))]

    total_images = 0
    class_counts = {}

    for class_name in sorted(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        count = len(image_files)
        class_counts[class_name] = count
        total_images += count
        print(f"  {class_name}: {count} images")

    print(f"\nTotal images: {total_images}")
    print(f"Number of classes: {len(class_dirs)}")

    if total_images > 0:
        print(f"Average per class: {total_images / len(class_dirs):.1f}")
        print(f"Min class size: {min(class_counts.values())}")
        print(f"Max class size: {max(class_counts.values())}")

    print("="*60 + "\n")

    return class_counts


# ============================================================================
# Helper Functions
# ============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def list_models(model_dir=None):
    """
    List all saved models in model directory

    Args:
        model_dir: Path to model directory

    Returns:
        dict: Model paths and sizes
    """
    model_dir = model_dir or config.MODEL_DIR

    print("\n" + "="*60)
    print(f"Saved Models in {model_dir}")
    print("="*60)

    if not os.path.exists(model_dir):
        print("Model directory not found")
        return {}

    models = {}

    for file in sorted(os.listdir(model_dir)):
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path):
            size_mb = get_model_size(file_path)
            models[file] = size_mb
            print(f"  {file}: {size_mb:.2f} MB")

    print("="*60 + "\n")

    return models


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    config.create_directories()

    if os.path.exists(config.TRAIN_DIR):
        analyze_dataset(config.TRAIN_DIR)

    if os.path.exists(config.VAL_DIR):
        analyze_dataset(config.VAL_DIR)

    list_models()
