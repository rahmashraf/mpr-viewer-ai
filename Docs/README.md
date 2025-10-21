# MPR Viewer with Segmentation
**Multi-Planar Reconstruction Viewer for Medical Imaging**
# Overview

**A comprehensive medical imaging visualization tool for DICOM and NIfTI files**

MPR Viewer is a powerful, user-friendly desktop application designed for medical professionals and researchers to visualize and analyze scans. Built with Python and PyQt5, it provides an intuitive interface for viewing medical images in multiple orientations, with advanced features for segmentation analysis, ROI extraction, and AI-powered orientation detection.

---

# Features

- ### Three Viewports:
  Simultaneously view Axial, Coronal, and Sagittal slices.
 ![Multi-Planar Views](path/to/mpr-views.gif)
- ### Visualization Controls <br>
   **9 Colormap Options**: gray, viridis, plasma, inferno, magma, cividis, jet, hot, cool <br>
   **Brightness/Contrast Adjustment** <br>
![Visualization Controls](path/to/visualization-controls.gif)
- ### Crosshair Navigation:
  Automatically synchronize slice navigation across all planes.
- ### Cine Mode:
  Play slices as an animated sequence.
- ### Mouse Interaction:
  Interact with the images directly using the mouse.
  
- ### AI Orientation Detection
Automatically detect scan orientation with confidence scores:
- Supports Axial, Coronal, and Sagittal detection
- Deep learning model with high accuracy
- Arabic and English labels for medical professionals

![Orientation Detection](path/to/orientation-detection.gif)

- ### Segmentation Analysis
- Load and visualize segmentation masks
- **Outline Mode**: Display segmentation boundaries on scan
- Automatic resampling for mismatched dimensions
- Multi-label support

![Segmentation Overlay](path/to/segmentation-overlay.gif)

- ### Region of Interest (ROI) Tools
- Interactive ROI drawing on any view
- 3D ROI propagation across all slices
- Volume extraction and NIfTI export
- Real-time preview with cyan highlighting

![ROI Drawing](path/to/roi-drawing.gif)

- ### Oblique View 
- Custom oblique plane rendering
- Adjustable rotation angles (X & Y axes)
- Real-time 3D transformation
- Independent slice navigation

![Oblique View](path/to/oblique-view.gif)

---

# Tech Stack

### Core Technologies
| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Primary programming language |
| ![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white) | GUI framework |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | Array processing |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge) | Visualization engine |

### Medical Imaging Libraries
- **SimpleITK** - Medical image I/O and processing
- **pydicom** - DICOM file handling
- **scikit-image** - Image processing algorithms

### Deep Learning
- **TensorFlow/Keras** - Orientation detection model
- Custom CNN architecture for medical image classification

### Additional Tools
- **SciPy** - Scientific computing utilities
- **Qt5 Styling** - Modern dark-themed UI

To install all required libraries, run:

```bash
pip install PyQt5 matplotlib SimpleITK vtk numpy scipy scikit-image pydicom tensorflow
```
# Installation and Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/mpr-viewer.git
   cd mpr-viewer
   ```

2. Run the application:

   ```bash
   python test 101.py
   ```

---

## Supported File Formats

- **NIfTI**: `.nii`, `.nii.gz`
- Directory-based DICOM series are supported.

---


## License

This project is licensed under the MIT License. See the [LICENSE](License) file for details.
---

## Contribution



---




