# InstanceSketch: Robust Sketch Instance Segmentation with Cluster Refinement

This repository contains the implementation of InstanceSketch, a two-step pipeline for stroke segmentation in free-hand sketches. The method integrates CNNs and Transformers to efficiently segment sketches into interpretable components, with a novel clustering-based refinement technique for distinguishing similar components.

## Overview

InstanceSketch addresses the challenge of stroke segmentation in hand-drawn sketches through a two-stage approach:
1. **Initial Stroke Segmentation**: Uses CNN + Transformer architecture to assign initial labels to strokes
2. **Stroke Label Refinement**: Employs clustering techniques to distinguish between similar components (e.g., left vs right legs)

## Features

- **Two-Stage Pipeline**: Initial segmentation followed by clustering-based refinement
- **Multi-Class Support**: Handles segmentation of different object classes simultaneously
- **Scene Segmentation**: InstanceSketch-Scene extension for multi-object scenes
- **YOLO Integration**: Generates datasets for object detection training
- **Efficient Architecture**: Lightweight model suitable for real-time applications

## Model Architecture

InstanceSketch consists of two main components (current settings can be changed):

### 1. Segmentation Model
- **Architecture**: CNN + Transformer encoder
- **Input**: Two NxN stroke images (shape and spatial representations)
- **Output**: Initial stroke labels
- **Features**: Dual-branch CNN, 2-layer Transformer, class-weighted loss

### 2. Refinement Model
- **Purpose**: Clustering-based refinement for similar components
- **Architecture**: CNN + Transformer for stroke embeddings
- **Usage**: Distinguishes between similar parts (e.g., left vs right legs)
- **Training**: Triplet loss for clustering

## Dataset

Dataset can be loaded in 2 separate formats (.xml or .json).
The model is currently configured to support 7 sketch categories:
"car", "cloud", "flower", "sun", "tree", "4leg", "2leg"

## Performance

- **Segmentation Accuracy**: 93.31%
- **F1-Score**: 90.50%
- **Model Size**: ~0.9M parameters
- **Inference Time**: 0.0029s per sample

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd instance-sketch
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- PyTorch
- ClearML (for experiment logging)
- Matplotlib
- Scikit-learn
- TQDM
- ultralytics

## Usage

### Training the Model

Train the stroke segmentation model:

Parameters for model training (dataset location, model parameters, training parameters, etc.) can be set in `opts.py`
```bash
python main.py
```

To use InstanceSketch-Scene, the YOLO detection model should be trained separately. To generate the dataset, use the following command:

### Generating YOLO Dataset

Generate dataset for object detection training:

```bash
python generate_detector_dataset.py
```

### Training YOLO Detector

Train YOLO model on the generated dataset:

```bash
yolo train data=path_to_generated_dataset/data.yaml model=yolov8n.pt epochs=150
```

### Running Inference

Run inference on sketch scenes:

```bash
python run_detector_inference.py
```

## License

This project is licensed under the Apache License 2.0.
