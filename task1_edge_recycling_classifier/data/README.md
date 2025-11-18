# Dataset Information

## Overview

This directory contains the training, validation, and test datasets for the Edge AI Recyclable Classifier.

## Required Structure

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
├── val/
│   ├── plastic/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   └── other/
├── test/ (optional)
│   └── (same structure as train)
└── sample_images/
    └── (test images for inference)
```

## Dataset Sources

### Recommended Public Datasets

1. **TrashNet Dataset**
   - Source: https://github.com/garythung/trashnet
   - Size: ~2,500 images
   - Classes: 6 (glass, paper, cardboard, plastic, metal, trash)
   - License: Open source

2. **Waste Classification Data (Kaggle)**
   - Source: https://www.kaggle.com/datasets/techsash/waste-classification-data
   - Size: 25,000+ images
   - Classes: Organic and recyclable waste
   - License: CC0: Public Domain

3. **RecycleNet**
   - Various recyclable items datasets on Kaggle
   - Search: "recyclable waste classification"

### Custom Dataset Guidelines

If creating your own dataset:

**Minimum Requirements:**
- At least 50-100 images per class
- Recommended: 200-500 images per class
- Use 80/20 or 70/30 train/validation split

**Image Quality:**
- Resolution: At least 224×224 pixels (will be resized)
- Format: JPG, PNG
- Lighting: Varied conditions
- Background: Mixed backgrounds for better generalization
- Angles: Multiple viewpoints of each object

**Class Balance:**
- Try to keep similar numbers of images per class
- If imbalanced, consider data augmentation or class weights

## Downloading and Preparing Data

### Option 1: Download TrashNet

```bash
# Clone TrashNet repository
git clone https://github.com/garythung/trashnet.git

# Organize into train/val folders (manual or use script below)
python scripts/prepare_trashnet.py
```

### Option 2: Kaggle Dataset

1. Install Kaggle API:
```bash
pip install kaggle
```

2. Download dataset:
```bash
kaggle datasets download -d techsash/waste-classification-data
unzip waste-classification-data.zip -d data/raw/
```

3. Organize into required structure:
```bash
python scripts/organize_dataset.py --input data/raw/ --output data/
```

### Option 3: Create Sample Dataset

For quick testing, create a small sample dataset:

```python
# scripts/create_sample_dataset.py
import os
from pathlib import Path

classes = ['plastic', 'glass', 'metal', 'paper', 'other']

for split in ['train', 'val']:
    for class_name in classes:
        path = Path(f'data/{split}/{class_name}')
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")

print("\nNow add at least 20-30 images per class in each folder")
```

## Dataset Statistics

After organizing your dataset, analyze it:

```python
import sys
sys.path.append('./src')
import utils

utils.analyze_dataset('data/train')
utils.analyze_dataset('data/val')
```

## Data Augmentation

The training pipeline automatically applies augmentation:
- Random rotation (±20°)
- Random zoom (±15%)
- Random horizontal flip
- Random width/height shift (±20%)
- Random shear transformation

## Sample Images

Place a few test images in `data/sample_images/` for quick inference testing.

## Notes

- Large image files are not committed to git (see .gitignore)
- Keep total dataset size reasonable (<1GB for quick training)
- For production models, use larger datasets (1000+ per class)
- Consider data quality over quantity

## Troubleshooting

**Issue**: "Found 0 images" error
- **Solution**: Check folder structure matches expected format
- Ensure images have .jpg, .jpeg, or .png extensions

**Issue**: Low accuracy
- **Solution**: Increase dataset size, ensure balanced classes, check image quality

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in config.py or use smaller input images
