# MPR Viewer with Segmentation

## Description

**A comprehensive medical imaging visualization tool for DICOM and NIfTI files**

MPR Viewer is a powerful, user-friendly desktop application designed for medical professionals and researchers to visualize and analyze scans. Built with Python and PyQt5, it provides an intuitive interface for viewing medical images in multiple orientations, with advanced features for segmentation analysis, ROI extraction, and AI-powered orientation detection.

---

## Features

- **Three Viewports**: Simultaneously view Axial, Coronal, and Sagittal slices.
- **Volume Rendering**: Visualize 3D volume data with interactive zoom and rotation.
- **Colormap Selection**: Choose from multiple colormaps for better visualization (e.g., Gray, Viridis, Plasma, Jet).
- **Brightness & Contrast Adjustment**: Fine-tune image visibility for each plane.
- **Crosshair Navigation**: Automatically synchronize slice navigation across all planes.
- **Cine Mode**: Play slices as an animated sequence.
- **Mouse Interaction**: Zoom, pan, and interact with the images directly using the mouse.
- **Reset View**: Instantly reset brightness, contrast, and crosshair positions to default.

---

## Prerequisites

Ensure you have the following dependencies installed on your system:

- **Python 3.8+**
- Libraries:
  - PyQt5
  - Matplotlib
  - SimpleITK
  - NumPy
  - VTK

To install the required libraries, run:

```bash
pip install PyQt5 matplotlib SimpleITK vtk numpy
```

---

## Installation and Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/mpr-viewer.git
   cd mpr-viewer
   ```

2. Run the application:

   ```bash
   python MPR_Viewer1.py
   ```

3. Use the **Load MRI Scan** button to load NIfTI (`.nii` or `.nii.gz`) files into the viewer.

4. Explore features such as:

   - Adjusting brightness and contrast for better visualization.
   - Navigating slices with sliders or mouse clicks.
   - Using the **Play/Pause** button to view slices in motion.
   - Selecting colormaps from the dropdown menu.
   - panning using arrow keys

---

## Supported File Formats

- **NIfTI**: `.nii`, `.nii.gz`
- Directory-based DICOM series are also supported.

---
## Slice View

The MRI scan can be viewed in multiple slices. Below is an example of a slice view:

![Axial slice View](MPR-Viewer\Docs\Brain_Views\axial_slice.jpg)
---

## Screenshots

![Axial, Coronal, and Sagittal slices in synchronized crosshair mode](MPR-Viewer\screenshots\Screenshot3.png)

![Axial, Coronal, and Sagittal slices in synchronized crosshair mode in addition to the 3d volume](MPR-Viewer\screenshots\Screenshot1.png)
---

## Future Enhancements

- Incorporate segmentation overlays for enhanced analysis.
- Add support for additional medical imaging formats.
- Improve volume rendering with advanced transfer functions.
---

## License

This project is licensed under the MIT License. See the [LICENSE](License) file for details.
---

## Contribution

Contributions are welcome! Please open an issue or submit a pull request to help improve the MPR Viewer.

---

## Project Demo

[Watch the video](https://github.com/user-attachments/assets/698139ca-6b05-4109-9e5c-25af1b6d4f48)

---

## Acknowledgements

- **PyQt5** for the responsive user interface.
- **SimpleITK** for efficient medical image processing.
- **VTK** for interactive 3D volume rendering.
- **Matplotlib** for intuitive visualization in the 2D views.

