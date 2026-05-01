# Temporal and Spatial Alignment for Sensor Fusion

## Overview

This project investigates how spatial and temporal alignment affect LiDAR–radar sensor fusion for autonomous vehicle perception using the nuScenes dataset.

The work focuses on object-level fusion between LiDAR detections and radar measurements, with particular attention to:

* object association
* velocity estimation
* temporal tracking consistency
* alignment degradation and recovery

A simplified and interpretable fusion pipeline is used to isolate the effect of alignment errors rather than building a full production fusion system.

This repository contains the main experimental code used for the dissertation:

**Temporal and Spatial Alignment for Sensor Fusion**

---

## Main Features

### Baseline Fusion

* LiDAR bounding boxes used as spatial references
* Radar points associated using point-in-box matching
* Velocity estimated from associated radar measurements
* Constant-velocity Kalman filter for track evaluation

### Spatial Degradation Testing

* translation offsets
* yaw rotation errors
* fusion sensitivity analysis

### Temporal Degradation Testing

* frame-level offsets
* timestamp corruption
* drift simulation
* jitter and mismatch analysis

### Recovery Methods

* overlap-based correction
* physically constrained scoring
* continuity-aware temporal correction
* interpolation
* Kalman filtering
* multiframe fusion
* constant-acceleration tracking comparison

---

## Technologies Used

* Python 3.11.9
* NumPy
* Pandas
* Matplotlib
* nuScenes SDK

---

## Dataset

This project uses the **nuScenes dataset**, a multimodal autonomous driving dataset containing camera, LiDAR, radar, and annotated object data.

Dataset link:

https://www.nuscenes.org/

Official dataset paper:

Caesar, H. et al. (2020) *nuScenes: A Multimodal Dataset for Autonomous Driving*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11621–11631.

---

## Running the Project

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib nuscenes-devkit
```

### 2. Download nuScenes dataset

Download and extract the dataset from:

https://www.nuscenes.org/

Update the dataset path inside the scripts/notebooks.

Example:

```python
nusc = NuScenes(
    version='v1.0-mini',
    dataroot='your_dataset_path',
    verbose=True
)
```

### 3. Run experiments

Main experiments include:

* baseline fusion evaluation
* temporal degradation testing
* spatial degradation testing
* alignment correction experiments
* multiframe fusion comparison

These can be run through the provided notebook or Python scripts (there isn't any yet).

---

## Dissertation Context

This repository supports a Master's dissertation focused on:

**Temporal and Spatial Alignment for Sensor Fusion**

The work forms part of a wider team project involving:

* camera perception
* LiDAR processing
* radar processing
* uncertainty-aware fusion
* autonomous vehicle perception systems

This repository specifically focuses on the alignment stage:

**Sensors => Alignment => Fusion => Output**

---

## Notes

The code included here contains the main runnable components used to generate the reported results.

Some experimental notebooks, temporary testing scripts, and intermediate development files are intentionally excluded to keep the repository focused and readable.

---

## Author

NikTop

MEng Robotics Engineering

Queen Mary University of London
