"""
Configuration file for Edge AI Recyclable Classifier
Centralized hyperparameters and settings
"""

import os

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ============================================================================
# Dataset Configuration
# ============================================================================
# Class names for recyclable items
CLASS_NAMES = ['glass', 'metal', 'paper', 'plastic', 'other']
NUM_CLASSES = len(CLASS_NAMES)

# Data directories
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, 'sample_images')

# ============================================================================
# Model Architecture
# ============================================================================
# Input image size (height, width)
IMG_SIZE = (160, 160)  # Options: (96, 96), (128, 128), (160, 160), (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)  # Add channels dimension

# Base model configuration
BASE_MODEL = 'MobileNetV2'  # Options: 'MobileNetV2', 'MobileNetV3Small', 'EfficientNetB0'
USE_IMAGENET_WEIGHTS = True
FREEZE_BASE_MODEL = True  # Set False for fine-tuning

# Classification head
DROPOUT_RATE = 0.3
FINAL_ACTIVATION = 'softmax'

# ============================================================================
# Training Hyperparameters
# ============================================================================
BATCH_SIZE = 32
EPOCHS = 15
INITIAL_LEARNING_RATE = 0.001

# Optimizer
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'rmsprop'

# Loss function
LOSS_FUNCTION = 'categorical_crossentropy'

# Metrics
METRICS = ['accuracy']

# ============================================================================
# Data Augmentation
# ============================================================================
# Training augmentation parameters
AUGMENTATION_CONFIG = {
    'rescale': 1./255,
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Validation/test augmentation (only rescaling)
VAL_AUGMENTATION_CONFIG = {
    'rescale': 1./255
}

# ============================================================================
# Training Callbacks
# ============================================================================
# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MONITOR = 'val_loss'

# Model checkpoint
SAVE_BEST_ONLY = True
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'

# Reduce learning rate on plateau
USE_REDUCE_LR = True
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_LR = 1e-7

# ============================================================================
# Model Saving
# ============================================================================
# Keras model
KERAS_MODEL_NAME = 'recyclable_classifier.keras'
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, KERAS_MODEL_NAME)

# SavedModel format (for TFLite conversion)
SAVED_MODEL_DIR = os.path.join(MODEL_DIR, 'saved_model')

# TFLite models
TFLITE_MODEL_NAME = 'recyclable_classifier.tflite'
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, TFLITE_MODEL_NAME)

TFLITE_FLOAT16_NAME = 'recyclable_classifier_float16.tflite'
TFLITE_FLOAT16_PATH = os.path.join(MODEL_DIR, TFLITE_FLOAT16_NAME)

TFLITE_INT8_NAME = 'recyclable_classifier_int8.tflite'
TFLITE_INT8_PATH = os.path.join(MODEL_DIR, TFLITE_INT8_NAME)

# ============================================================================
# TFLite Conversion
# ============================================================================
# Quantization settings
ENABLE_FLOAT16_QUANTIZATION = True
ENABLE_INT8_QUANTIZATION = True

# Representative dataset size for INT8 quantization
REPRESENTATIVE_DATASET_SIZE = 100

# ============================================================================
# Inference Configuration
# ============================================================================
# Default model for inference
DEFAULT_INFERENCE_MODEL = TFLITE_INT8_PATH

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# Results and Logging
# ============================================================================
# Training history
HISTORY_JSON_PATH = os.path.join(RESULTS_DIR, 'training_history.json')

# Plots
TRAINING_PLOTS_DIR = os.path.join(RESULTS_DIR, 'training_plots')
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')

# Performance metrics
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, 'performance_metrics.json')

# ============================================================================
# Hardware Configuration
# ============================================================================
# Use GPU if available
USE_GPU = True

# Mixed precision training (faster on compatible GPUs)
USE_MIXED_PRECISION = False

# TensorFlow logging level
TF_LOG_LEVEL = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# ============================================================================
# Raspberry Pi Deployment
# ============================================================================
# Target edge device specifications
EDGE_DEVICE = 'raspberry_pi_4'  # Options: 'raspberry_pi_4', 'jetson_nano', 'coral_dev'

# Expected latency targets (ms)
TARGET_LATENCY_MS = 50

# ============================================================================
# Helper Functions
# ============================================================================
def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
    print("✓ Directories created/verified")

def get_device_info():
    """Get information about available devices"""
    import tensorflow as tf

    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"  - {gpu}")

    print("=" * 60)

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"Classes: {CLASS_NAMES}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {INITIAL_LEARNING_RATE}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print("=" * 60)

# ============================================================================
# Optional: Advanced Configurations
# ============================================================================

# Fine-tuning configuration (if unfreezing base model)
FINE_TUNE_AT = 100  # Unfreeze layers from this index onwards
FINE_TUNE_LEARNING_RATE = 1e-5

# Test-time augmentation (TTA)
USE_TTA = False
TTA_STEPS = 5

# Class weights (for imbalanced datasets)
USE_CLASS_WEIGHTS = False
CLASS_WEIGHTS = None  # Will be computed if needed

if __name__ == "__main__":
    create_directories()
    get_device_info()
    print_config()
