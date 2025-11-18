"""
Convert trained Keras model to TensorFlow Lite format
Supports float32, float16, and INT8 quantization
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(__file__))
import config
import utils


def convert_to_tflite_float32(model_path, output_path):
    """
    Convert model to TFLite (float32)

    Args:
        model_path: Path to saved Keras model or SavedModel directory
        output_path: Output path for TFLite model

    Returns:
        Path to saved TFLite model
    """
    print("\n" + "="*60)
    print("Converting to TFLite (Float32)")
    print("="*60)

    # Load model
    if os.path.isdir(model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = utils.get_model_size(output_path)
    print(f"✓ Float32 model saved to {output_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print("="*60)

    return output_path


def convert_to_tflite_float16(model_path, output_path):
    """
    Convert model to TFLite with float16 quantization

    Args:
        model_path: Path to saved Keras model or SavedModel directory
        output_path: Output path for TFLite model

    Returns:
        Path to saved TFLite model
    """
    print("\n" + "="*60)
    print("Converting to TFLite (Float16 Quantization)")
    print("="*60)

    # Load model
    if os.path.isdir(model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply float16 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = utils.get_model_size(output_path)
    print(f"✓ Float16 model saved to {output_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print("="*60)

    return output_path


def convert_to_tflite_int8(model_path, output_path, representative_dataset_fn=None):
    """
    Convert model to TFLite with INT8 quantization

    Args:
        model_path: Path to saved Keras model or SavedModel directory
        output_path: Output path for TFLite model
        representative_dataset_fn: Function that yields representative data

    Returns:
        Path to saved TFLite model
    """
    print("\n" + "="*60)
    print("Converting to TFLite (INT8 Quantization)")
    print("="*60)

    # Load model
    if os.path.isdir(model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_dataset_fn:
        converter.representative_dataset = representative_dataset_fn
        # Full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        print("Using full integer quantization with representative dataset")
    else:
        print("Using dynamic range quantization (no representative dataset)")

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = utils.get_model_size(output_path)
    print(f"✓ INT8 model saved to {output_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print("="*60)

    return output_path


def create_representative_dataset(data_dir=None, num_samples=100):
    """
    Create representative dataset generator for INT8 quantization

    Args:
        data_dir: Path to image directory
        num_samples: Number of samples to use

    Returns:
        Generator function
    """
    data_dir = data_dir or config.TRAIN_DIR

    # Create data generator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=config.IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=True
    )

    def representative_dataset_gen():
        for i in range(num_samples):
            img, _ = next(generator)
            yield [img.astype(np.float32)]

    return representative_dataset_gen


def compare_models(models_dict):
    """
    Compare sizes of different model formats

    Args:
        models_dict: Dictionary of {name: path}
    """
    print("\n" + "="*60)
    print("Model Size Comparison")
    print("="*60)

    sizes = {}
    for name, path in models_dict.items():
        if os.path.exists(path):
            size_mb = utils.get_model_size(path)
            sizes[name] = size_mb
            print(f"{name:20s}: {size_mb:8.2f} MB")
        else:
            print(f"{name:20s}: Not found")

    if sizes:
        baseline = list(sizes.values())[0]
        print("\n" + "-"*60)
        print("Size Reduction:")
        for name, size in sizes.items():
            reduction = (1 - size/baseline) * 100
            print(f"{name:20s}: {reduction:6.1f}% reduction")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to TFLite')
    parser.add_argument('--model_path', type=str, default=config.SAVED_MODEL_DIR,
                        help='Path to Keras model or SavedModel directory')
    parser.add_argument('--output_dir', type=str, default=config.MODEL_DIR,
                        help='Output directory for TFLite models')
    parser.add_argument('--data_dir', type=str, default=config.TRAIN_DIR,
                        help='Training data directory for representative dataset')
    parser.add_argument('--formats', type=str, default='all',
                        choices=['all', 'float32', 'float16', 'int8'],
                        help='Which formats to convert to')
    parser.add_argument('--representative_samples', type=int, default=100,
                        help='Number of samples for INT8 quantization')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("TensorFlow Lite Conversion Tool")
    print("="*60)
    print(f"Input model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Formats: {args.formats}")
    print("="*60)

    models = {}

    # Convert to float32
    if args.formats in ['all', 'float32']:
        output_path = os.path.join(args.output_dir, 'recyclable_classifier.tflite')
        convert_to_tflite_float32(args.model_path, output_path)
        models['Float32'] = output_path

    # Convert to float16
    if args.formats in ['all', 'float16']:
        output_path = os.path.join(args.output_dir, 'recyclable_classifier_float16.tflite')
        convert_to_tflite_float16(args.model_path, output_path)
        models['Float16'] = output_path

    # Convert to INT8
    if args.formats in ['all', 'int8']:
        output_path = os.path.join(args.output_dir, 'recyclable_classifier_int8.tflite')

        # Create representative dataset if training data exists
        representative_dataset_fn = None
        if os.path.exists(args.data_dir):
            print(f"Creating representative dataset from {args.data_dir}...")
            representative_dataset_fn = create_representative_dataset(
                args.data_dir,
                args.representative_samples
            )

        convert_to_tflite_int8(args.model_path, output_path, representative_dataset_fn)
        models['INT8'] = output_path

    # Compare model sizes
    if len(models) > 1:
        compare_models(models)

    print("\n✓ Conversion complete!")
    print("\nNext steps:")
    print("1. Test inference using src/inference.py")
    print("2. Benchmark latency on target device")
    print("3. Deploy to Raspberry Pi or edge device")


if __name__ == "__main__":
    main()
