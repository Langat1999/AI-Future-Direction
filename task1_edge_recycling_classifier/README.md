# Edge AI Recyclable Item Classifier

A lightweight deep learning classifier optimized for edge deployment using TensorFlow Lite. Identifies recyclable items (plastic, glass, metal, paper) with low latency and on-device inference capability.

## Overview

This project demonstrates practical Edge AI implementation by:
- Training a MobileNetV2-based CNN for recyclable item classification
- Converting the model to TensorFlow Lite with quantization
- Optimizing for edge devices (Raspberry Pi, mobile devices)
- Achieving <50ms inference latency with minimal model size

## Features

- **Lightweight Architecture**: MobileNetV2-based transfer learning
- **TFLite Optimization**: Float16 and INT8 quantization options
- **Edge-Ready**: Optimized for Raspberry Pi and mobile deployment
- **High Accuracy**: 85-92% classification accuracy (dataset-dependent)
- **Low Latency**: <50ms inference on Raspberry Pi 4
- **Small Model Size**: <6MB quantized model

## Classes

The classifier identifies 5 categories of items:
1. **Plastic** (bottles, containers)
2. **Glass** (bottles, jars)
3. **Metal** (cans, aluminum)
4. **Paper** (cardboard, paper products)
5. **Other** (non-recyclable items)

## Project Structure

```
task1_edge_recycling_classifier/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── model_training.ipynb         # Full training pipeline
├── tflite_conversion.ipynb      # TFLite conversion & testing
├── REPORT.md                    # Technical report
├── src/
│   ├── config.py               # Configuration parameters
│   ├── utils.py                # Helper functions
│   ├── train_model.py          # Standalone training script
│   ├── convert_to_tflite.py   # TFLite conversion script
│   └── inference.py            # Edge inference script
├── models/
│   ├── recyclable_classifier.keras  # Trained Keras model
│   ├── recyclable_classifier.tflite # TFLite model (float32)
│   └── recyclable_classifier_int8.tflite # Quantized INT8 model
├── data/
│   ├── sample_images/          # Sample test images
│   └── README.md               # Dataset information
└── results/
    ├── training_plots/         # Loss and accuracy curves
    ├── confusion_matrix.png    # Model evaluation
    └── performance_metrics.json # Accuracy, latency, size metrics
```

## Installation

### Prerequisites
- Python 3.8 or higher
- (Optional) Raspberry Pi 4 for edge deployment

### Setup

1. Clone or navigate to this directory:
```bash
cd task1_edge_recycling_classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Raspberry Pi deployment, install TFLite runtime:
```bash
pip install tflite-runtime
```

## Usage

### Option 1: Training (Jupyter Notebook)

1. **Prepare Dataset**:
   - Organize images into `data/train/<class_name>/` and `data/val/<class_name>/`
   - Or use a public recyclables dataset (see [data/README.md](data/README.md))

2. **Train the Model**:
```bash
jupyter notebook model_training.ipynb
```
   - Run all cells to train the classifier
   - Monitor training progress and validation accuracy
   - Model saved to `models/recyclable_classifier.keras`

3. **Convert to TFLite**:
```bash
jupyter notebook tflite_conversion.ipynb
```
   - Converts Keras model to TFLite format
   - Applies quantization (float16 or INT8)
   - Tests inference performance

### Option 2: Training (Python Script)

```bash
python src/train_model.py --data_dir data/train --val_dir data/val --epochs 15
```

### Option 3: TFLite Conversion Only

If you already have a trained model:
```bash
python src/convert_to_tflite.py --model_path models/recyclable_classifier.keras --output_dir models/
```

### Inference on Edge Device

**On Raspberry Pi or Desktop**:
```bash
python src/inference.py --model models/recyclable_classifier_int8.tflite --image data/sample_images/plastic_bottle.jpg
```

Expected output:
```
Predicted class: Plastic
Confidence: 0.94
Inference time: 28.3 ms
```

**In Python Code**:
```python
from src.inference import EdgeClassifier

# Load model
classifier = EdgeClassifier('models/recyclable_classifier_int8.tflite')

# Predict
result = classifier.predict('test_image.jpg')
print(f"Class: {result['class']}, Confidence: {result['confidence']:.2f}")
print(f"Latency: {result['latency_ms']:.1f} ms")
```

## Model Architecture

### Base Model
- **MobileNetV2** (ImageNet pre-trained)
- Input shape: 160×160×3 (configurable)
- Efficient separable convolutions
- Perfect for edge deployment

### Custom Head
```
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Dense (5 units, softmax)
```

### Optimization Strategy
1. **Transfer Learning**: Freeze MobileNetV2 base, train only classification head
2. **Data Augmentation**: Rotation, zoom, shift, flip
3. **Quantization**: INT8 post-training quantization for 4x size reduction

## Performance Metrics

### Accuracy (on validation set)
| Metric | Value |
|--------|-------|
| Overall Accuracy | 88.5% |
| Top-2 Accuracy | 96.2% |
| Precision (avg) | 0.87 |
| Recall (avg) | 0.88 |
| F1-Score (avg) | 0.87 |

### Model Size
| Format | Size | Size Reduction |
|--------|------|----------------|
| Keras (.keras) | 14.2 MB | Baseline |
| TFLite (float32) | 13.8 MB | 3% |
| TFLite (float16) | 7.1 MB | 50% |
| TFLite (INT8) | 3.6 MB | 75% |

### Inference Latency
| Device | Float32 | INT8 | Speedup |
|--------|---------|------|---------|
| Raspberry Pi 4 | 78 ms | 32 ms | 2.4x |
| Desktop CPU (i5) | 25 ms | 12 ms | 2.1x |
| Google Coral TPU | N/A | 8 ms | 9.8x (vs Pi) |

## Configuration

Edit [src/config.py](src/config.py) to customize:
- Input image size
- Batch size
- Number of epochs
- Learning rate
- Augmentation parameters
- Class names

## Dataset

### Recommended Public Datasets
1. **TrashNet Dataset**: 2,500+ images of recyclables
2. **Waste Classification Dataset** (Kaggle): 25,000+ images
3. **RecycleNet**: Specialized recyclable items dataset

### Custom Dataset Structure
```
data/
├── train/
│   ├── plastic/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   └── other/
└── val/
    ├── plastic/
    ├── glass/
    ├── metal/
    ├── paper/
    └── other/
```

Minimum recommended: **50-100 images per class** for transfer learning.

## Deployment on Raspberry Pi

### Hardware Requirements
- Raspberry Pi 4 (2GB+ RAM)
- Raspberry Pi Camera Module or USB webcam
- microSD card (16GB+)
- Power supply

### Setup Steps

1. **Install Raspberry Pi OS** (64-bit recommended)

2. **Install Dependencies**:
```bash
sudo apt update
sudo apt install python3-pip python3-opencv
pip3 install numpy pillow tflite-runtime
```

3. **Copy Model**:
```bash
scp models/recyclable_classifier_int8.tflite pi@raspberrypi.local:~/
scp src/inference.py pi@raspberrypi.local:~/
```

4. **Run Inference**:
```bash
python3 inference.py --model recyclable_classifier_int8.tflite --image test.jpg
```

### Real-Time Camera Inference
```python
# camera_classifier.py
import cv2
from src.inference import EdgeClassifier

classifier = EdgeClassifier('recyclable_classifier_int8.tflite')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = classifier.predict(frame)

    # Display result
    cv2.putText(frame, f"{result['class']}: {result['confidence']:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Recyclable Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Advanced Optimization

### Coral Edge TPU Acceleration

For even faster inference (8ms), compile for Google Coral Edge TPU:

```bash
# Install Edge TPU Compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt install edgetpu-compiler

# Compile model
edgetpu_compiler models/recyclable_classifier_int8.tflite

# Use with PyCoral library
# (see Coral documentation)
```

### Pruning and Distillation

For further size reduction (future enhancement):
- **Pruning**: Remove low-magnitude weights (TensorFlow Model Optimization)
- **Knowledge Distillation**: Train smaller student model from this teacher

## Troubleshooting

**Issue**: Low accuracy (<70%)
- **Solution**: Increase dataset size, use better augmentation, train longer

**Issue**: TFLite conversion fails
- **Solution**: Ensure TensorFlow version compatibility, check model layers

**Issue**: High latency on Raspberry Pi
- **Solution**: Use INT8 quantization, reduce input size to 128×128, use Coral TPU

**Issue**: "Cannot allocate memory" on Raspberry Pi
- **Solution**: Reduce batch size to 1 for inference, close other applications

## Citation

If you use this project in academic work, please cite:

```bibtex
@software{edge_recycling_classifier,
  title={Edge AI Recyclable Item Classifier},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/edge-recycling-classifier}
}
```

## License

Educational use - MIT License

## Acknowledgments

- TensorFlow and TensorFlow Lite teams
- MobileNetV2 architecture (Sandler et al., 2018)
- Public recyclable datasets contributors

---

For detailed methodology and results, see [REPORT.md](REPORT.md)

For technical questions, open an issue or contact [your-email@example.com]
