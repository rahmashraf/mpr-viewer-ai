# MPR Viewer with Segmentation
**Multi-Planar Reconstruction Viewer for Medical Imaging**
# Overview

**A comprehensive medical imaging visualization tool for DICOM and NIfTI files**

MPR Viewer is a powerful, user-friendly desktop application designed for medical professionals and researchers to visualize and analyze scans. Built with Python and PyQt5, it provides an intuitive interface for viewing medical images in multiple orientations, with advanced features for segmentation analysis, ROI extraction, and AI-powered orientation detection.

---

# Features

- ### Three Viewports:
  Simultaneously view Axial, Coronal, and Sagittal slices.
  
- ### Visualization Controls <br>
   **9 Colormap Options**: gray, viridis, plasma, inferno, magma, cividis, jet, hot, cool <br>
   **Brightness/Contrast Adjustment** <br>
   
 ![Visualization Controls](https://github.com/rahmashraf/mpr-viewer-ai/blob/main/assets/Visualization_controls.gif)
- ### Crosshair Navigation:
  Automatically synchronize slice navigation across all planes.
- ### Cine Mode:
  Play slices as an animated sequence.
- ### Mouse Interaction:
  Interact with the images directly using the mouse.
- ### AI Organ Detection
- The system integrates an AI-based model for automatic organ detection and classification within medical scans.
- Built using TensorFlow/Keras with a custom Convolutional Neural Network (CNN).
- Automatically identifies the primary organ or region in the scan (e.g., heart, brain, liver) with confidence scores.

![Oragn Detection](https://github.com/rahmashraf/mpr-viewer-ai/blob/main/assets/organ_detection.jpeg)

- ### AI Orientation Detection
  Automatically detect scan orientation with confidence scores:
- Supports Axial, Coronal, and Sagittal detection
- Deep learning model with high accuracy
- Arabic and English labels for medical professionals

 ![Orientation Detection](https://github.com/rahmashraf/mpr-viewer-ai/blob/main/assets/AI_orientation.gif)

- ### Segmentation Analysis
- Load and visualize segmentation masks
- **Outline Mode**: Display segmentation boundaries on scan
- Automatic resampling for mismatched dimensions
- Multi-label support

![Segmentation Overlay](https://github.com/rahmashraf/mpr-viewer-ai/blob/main/assets/Surface_outline.gif)

- ### Region of Interest (ROI) Tools
- Interactive ROI drawing on any view
- 3D ROI propagation across all slices
- Volume extraction and NIfTI export
- Real-time preview with cyan highlighting

![ROI Drawing](https://github.com/rahmashraf/mpr-viewer-ai/blob/main/assets/ROI_.gif)

- ### Oblique View 
- Custom oblique plane rendering
- Adjustable rotation angles (X & Y axes)
- Real-time 3D transformation
- Independent slice navigation

![Oblique View](https://github.com/rahmashraf/mpr-viewer-ai/blob/main/assets/Oblique_view.gif)

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


# Installation and Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/rahmashraf/mpr-viewer-ai.git
   ```

2. Install all required libraries, run:

   ```bash
   pip install PyQt5 matplotlib SimpleITK vtk numpy scipy scikit-image pydicom tensorflow
   ```

** The following files and folders are essential for the MPR Viewer to run correctly.
If any of them are missing, the application will not start or will throw errors.

## 📁 Project Structure

Please make sure your project structure matches the following:
```
src/
├── style.qss                     # Main style file for the application's dark theme
├── model/                        # Folder containing AI model and labels
│   ├── class_names.txt           # Organ class labels for detection
│   └── model.keras               # Trained deep learning model for organ/orientation detection
└── main/                         # Core application logic
    ├── main.py                   # Entry point of the application
    ├── detect_organ.py           # AI organ detection script
    └── detect_orientation.py     # Orientation detection module
```
These files are automatically loaded when you run main.py.
If any of them are missing or moved to a different location, please update the import paths in the code accordingly.
The application will not function properly without them.

3. Run the application:

   ```bash
   python main.py
   ```
   
## Supported File Formats

- **NIfTI**: `.nii`, `.nii.gz`
- Directory-based DICOM series are supported.

---


## License

This project is licensed under the MIT License. See the [LICENSE](License) file for details.
---






