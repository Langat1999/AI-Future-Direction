"""
TFLite inference script for Edge AI Recyclable Classifier
Supports both desktop and Raspberry Pi deployment
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Try to import tflite_runtime (Raspberry Pi) first, then fall back to tensorflow
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite_runtime (optimized for edge devices)")
except ImportError:
    import tensorflow.lite as tflite
    print("Using tensorflow.lite")

from PIL import Image

# Add src to path
sys.path.append(os.path.dirname(__file__))
import config


class EdgeClassifier:
    """
    Edge-optimized recyclable item classifier using TFLite
    """

    def __init__(self, model_path=None, class_names=None):
        """
        Initialize the classifier

        Args:
            model_path: Path to TFLite model file
            class_names: List of class names
        """
        self.model_path = model_path or config.DEFAULT_INFERENCE_MODEL
        self.class_names = class_names or config.CLASS_NAMES

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

        # Determine if model is quantized
        self.is_quantized = self.input_details[0]['dtype'] == np.uint8

        print(f"✓ Model loaded: {os.path.basename(self.model_path)}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Quantized: {self.is_quantized}")
        print(f"  Classes: {self.class_names}")

    def preprocess_image(self, image_input):
        """
        Preprocess image for inference

        Args:
            image_input: Path to image file, PIL Image, or numpy array

        Returns:
            Preprocessed image array
        """
        # Load image
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert('RGB')
        else:
            raise ValueError("Unsupported image input type")

        # Resize to model input size
        img = img.resize((self.input_width, self.input_height))

        # Convert to array
        img_array = np.array(img, dtype=np.float32)

        # Normalize and quantize if needed
        if self.is_quantized:
            # For quantized models (INT8)
            # Input is expected in range [0, 255] as uint8
            img_array = img_array.astype(np.uint8)
        else:
            # For float models, normalize to [0, 1]
            img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def postprocess_output(self, output):
        """
        Postprocess model output

        Args:
            output: Raw model output

        Returns:
            Processed predictions
        """
        # If quantized output, dequantize
        if self.output_details[0]['dtype'] == np.uint8:
            scale, zero_point = self.output_details[0]['quantization']
            output = scale * (output.astype(np.float32) - zero_point)

        return output

    def predict(self, image_input, return_all_probs=False):
        """
        Predict class for a single image

        Args:
            image_input: Image file path, PIL Image, or numpy array
            return_all_probs: Whether to return probabilities for all classes

        Returns:
            dict: Prediction results including class, confidence, and latency
        """
        # Preprocess
        img_array = self.preprocess_image(image_input)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)

        # Run inference with timing
        start_time = time.perf_counter()
        self.interpreter.invoke()
        end_time = time.perf_counter()

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        output = self.postprocess_output(output)

        # Get predictions
        probabilities = output[0]  # Remove batch dimension

        # Apply softmax if needed (some models output logits)
        if np.max(probabilities) > 1.0 or np.min(probabilities) < 0.0:
            exp_probs = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_probs / np.sum(exp_probs)

        # Get top prediction
        top_index = np.argmax(probabilities)
        top_class = self.class_names[top_index]
        confidence = float(probabilities[top_index])

        # Calculate latency
        latency_ms = (end_time - start_time) * 1000

        result = {
            'class': top_class,
            'class_index': int(top_index),
            'confidence': confidence,
            'latency_ms': latency_ms
        }

        if return_all_probs:
            result['all_probabilities'] = {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            }

        return result

    def predict_batch(self, image_inputs):
        """
        Predict classes for multiple images (processed sequentially)

        Args:
            image_inputs: List of image file paths or PIL Images

        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for img in image_inputs:
            result = self.predict(img)
            results.append(result)
        return results

    def benchmark(self, image_input, num_runs=100, warmup_runs=10):
        """
        Benchmark inference performance

        Args:
            image_input: Image for benchmarking
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs (not timed)

        Returns:
            dict: Benchmark statistics
        """
        print(f"\nBenchmarking with {num_runs} runs...")

        # Preprocess once
        img_array = self.preprocess_image(image_input)

        # Warmup
        for _ in range(warmup_runs):
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)

            start_time = time.perf_counter()
            self.interpreter.invoke()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        latencies = np.array(latencies)

        stats = {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'throughput_fps': float(1000 / np.mean(latencies))
        }

        print("\n" + "="*60)
        print("Benchmark Results")
        print("="*60)
        print(f"Mean latency:     {stats['mean_ms']:.2f} ms")
        print(f"Median latency:   {stats['median_ms']:.2f} ms")
        print(f"Std deviation:    {stats['std_ms']:.2f} ms")
        print(f"Min latency:      {stats['min_ms']:.2f} ms")
        print(f"Max latency:      {stats['max_ms']:.2f} ms")
        print(f"95th percentile:  {stats['p95_ms']:.2f} ms")
        print(f"99th percentile:  {stats['p99_ms']:.2f} ms")
        print(f"Throughput:       {stats['throughput_fps']:.1f} FPS")
        print("="*60)

        return stats


def main():
    parser = argparse.ArgumentParser(description='Run inference with TFLite model')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to TFLite model file')
    parser.add_argument('--image', type=str, required=False,
                        help='Path to input image')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark mode')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of benchmark runs')
    parser.add_argument('--all_probs', action='store_true',
                        help='Show probabilities for all classes')

    args = parser.parse_args()

    # Initialize classifier
    print("\n" + "="*60)
    print("Edge AI Recyclable Classifier - Inference")
    print("="*60)

    classifier = EdgeClassifier(model_path=args.model)

    # If no image provided, try to use a sample image
    if not args.image:
        sample_dir = config.SAMPLE_IMAGES_DIR
        if os.path.exists(sample_dir):
            # Find first image in sample_images
            for root, dirs, files in os.walk(sample_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        args.image = os.path.join(root, file)
                        print(f"\nUsing sample image: {args.image}")
                        break
                if args.image:
                    break

    if not args.image:
        print("\nError: No image provided and no sample images found.")
        print("Please specify an image using --image flag")
        sys.exit(1)

    # Benchmark mode
    if args.benchmark:
        classifier.benchmark(args.image, num_runs=args.runs)

    # Single prediction
    print("\n" + "="*60)
    print("Running Inference")
    print("="*60)

    result = classifier.predict(args.image, return_all_probs=args.all_probs)

    print(f"\nImage: {os.path.basename(args.image)}")
    print(f"Predicted Class: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"Inference Time: {result['latency_ms']:.2f} ms")

    if args.all_probs:
        print("\nAll Class Probabilities:")
        for class_name, prob in sorted(result['all_probabilities'].items(),
                                        key=lambda x: x[1], reverse=True):
            print(f"  {class_name:10s}: {prob:.4f} ({prob*100:.2f}%)")

    print("="*60)


if __name__ == "__main__":
    main()
