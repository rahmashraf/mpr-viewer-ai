import sys
import logging
from functools import wraps
from typing import Optional, Tuple, Any
import SimpleITK as sitk
import numpy as np
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
                             QWidget, QFileDialog, QSlider, QStatusBar, QGroupBox, QLabel,
                             QComboBox, QMessageBox, QRadioButton, QButtonGroup, QCheckBox,
                             QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor, QPalette, QColor
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pydicom
import os
from scipy.ndimage import rotate, zoom  # Removed binary_fill_holes
from skimage import measure  # Required for draw_surface_outline
from detect_orientation import predict_dicom_image





# ============================================================================
# ERROR HANDLING FRAMEWORK
# ============================================================================

class ErrorHandler:
    """Centralized error handling system"""

    @staticmethod
    def setup_logging():
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('mri_viewer.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('MRIViewer')

    @staticmethod
    def handle_error(error: Exception, context: str, severity: str = "error",
                     show_dialog: bool = True, parent: Optional[QWidget] = None) -> None:
        """
        Centralized error handling

        Args:
            error: The exception that occurred
            context: Description of what was being done when error occurred
            severity: 'critical', 'error', 'warning', or 'info'
            show_dialog: Whether to show a dialog to the user
            parent: Parent widget for dialog
        """
        logger = logging.getLogger('MRIViewer')

        error_msg = f"{context}: {str(error)}"

        # Log the error
        if severity == "critical":
            logger.critical(error_msg, exc_info=True)
        elif severity == "error":
            logger.error(error_msg, exc_info=True)
        elif severity == "warning":
            logger.warning(error_msg)
        else:
            logger.info(error_msg)

        # Show dialog if requested
        if show_dialog and parent:
            if severity in ["critical", "error"]:
                QMessageBox.critical(parent, "Error", f"{context}\n\n{str(error)}")
            elif severity == "warning":
                QMessageBox.warning(parent, "Warning", f"{context}\n\n{str(error)}")
            else:
                QMessageBox.information(parent, "Information", f"{context}\n\n{str(error)}")


def safe_execute(default_return=None, show_error=True):
    """Decorator for safe function execution with error handling"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                context = f"Error in {func.__name__}"
                ErrorHandler.handle_error(e, context, "error", show_error, self)
                return default_return

        return wrapper

    return decorator


def validate_array(func):
    """Decorator to validate numpy arrays before processing"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'scan_array') or self.scan_array is None:
            self.status_bar.showMessage("⚠ No scan loaded. Please load a scan first.", 5000)
            return None
        return func(self, *args, **kwargs)

    return wrapper


# ============================================================================
# MAIN MRI VIEWER CLASS
# ============================================================================

class MRIViewer(QWidget):
    def __init__(self):
        super().__init__()

        # Setup logging
        self.logger = ErrorHandler.setup_logging()
        self.logger.info("Initializing MRI Viewer")

        # Initialize data variables
        self.data = None
        self.scan_array = None
        self.sitk_image = None
        self.segmentation_array = None
        self.current_scan_path = None

        # Slice positions
        self.slices = [0, 0, 0]
        self.crosshair_x = 0
        self.crosshair_y = 0
        self.crosshair_z = 0

        # Marked points and zoom
        self.marked_points = [[], [], []]
        self.zoom_level = 1.0

        # Brightness/Contrast
        self.brightness = [0, 0, 0]
        self.contrast = [1.0, 1.0, 1.0]
        self.adjusting_window = False
        self.last_mouse_pos = None

        # View states
        self.panning = False
        self.pan_start = None
        self.current_colormap = 'gray'
        self.cine_running = False
        self.oblique_enabled = False
        # highlight-start
        self.outline_enabled = False  # New state for simple outline toggle
        # highlight-end

        # ROI variables
        self.drawing_roi = False
        self.roi_start_pos = None
        self.active_roi_view_index = -1
        self.roi_bounds_3d = None

        # Configure matplotlib
        self.setup_matplotlib()

        # Initialize UI
        self.initUI()

        self.logger.info("MRI Viewer initialized successfully")

    def setup_matplotlib(self):
        """Configure matplotlib styling"""
        plt.style.use('dark_background')
        matplotlib.rcParams.update({
            'figure.facecolor': '#2b2b2b',
            'axes.facecolor': '#2b2b2b',
            'savefig.facecolor': '#2b2b2b',
            'axes.edgecolor': '#404040',
            'axes.labelcolor': '#e0e0e0',
            'xtick.color': '#e0e0e0',
            'ytick.color': '#e0e0e0',
            'text.color': '#e0e0e0',
            'grid.color': '#404040',
            'figure.edgecolor': '#2b2b2b'
        })

    def initUI(self):
        """Initialize the user interface"""
        try:
            self.setWindowTitle("MRI Viewer - Professional Edition")
            self.setGeometry(100, 100, 1600, 900)

            # Apply modern styling
            self.setStyleSheet("""
                QWidget {
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 9pt;
                }
                QGroupBox {
                    border: 1px solid #404040;
                    border-radius: 6px;
                    margin-top: 12px;
                    padding-top: 15px;
                    font-weight: bold;
                    color: #00adb5;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QPushButton {
                    background-color: #2d2d30;
                    border: 1px solid #3e3e42;
                    border-radius: 4px;
                    padding: 6px 12px;
                    color: #e0e0e0;
                }
                QPushButton:hover {
                    background-color: #3e3e42;
                    border: 1px solid #00adb5;
                }
                QPushButton:pressed {
                    background-color: #007acc;
                }
                QPushButton:checked {
                    background-color: #00adb5;
                    color: #1e1e1e;
                }
                QPushButton:disabled {
                    background-color: #252526;
                    color: #6e6e6e;
                    border: 1px solid #2d2d30;
                }
                QComboBox {
                    background-color: #2d2d30;
                    border: 1px solid #3e3e42;
                    border-radius: 4px;
                    padding: 4px;
                }
                QComboBox:hover {
                    border: 1px solid #00adb5;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QSlider::groove:horizontal {
                    background: #3e3e42;
                    height: 6px;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #00adb5;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
                QSlider::handle:horizontal:hover {
                    background: #00d4dd;
                }
                QRadioButton {
                    spacing: 8px;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                    border-radius: 8px;
                    border: 2px solid #3e3e42;
                }
                QRadioButton::indicator:checked {
                    background-color: #00adb5;
                    border: 2px solid #00adb5;
                }
                QStatusBar {
                    background-color: #252526;
                    color: #e0e0e0;
                    border-top: 1px solid #3e3e42;
                }
                QLabel {
                    color: #e0e0e0;
                }
            """)

            self.main_layout = QHBoxLayout()
            self.create_control_panel()
            self.create_viewport_panel()

            self.main_layout.addWidget(self.control_panel)
            self.main_layout.addWidget(self.viewport_panel)

            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self.update_slices)
            self.is_playing = False

            self.setLayout(self.main_layout)
            self.setFocusPolicy(Qt.StrongFocus)

            self.logger.info("UI initialized successfully")

            self.detect_orientation_button = QPushButton("Detect Orientation", self)
            self.detect_orientation_button.clicked.connect(self.detect_orientation_action)
            self.control_layout.addWidget(self.detect_orientation_button)

        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to initialize UI", "critical", True, self)
            raise

    def create_control_panel(self):
        """Create the left control panel with all buttons"""
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        self.control_panel.setMaximumWidth(340)

        # === LOADING GROUP ===
        load_group = QGroupBox("📁 Load Data")
        load_layout = QVBoxLayout()

        self.load_nifti_button = QPushButton('📊 Load NIfTI Scan')
        self.load_nifti_button.clicked.connect(self.load_nifti)
        load_layout.addWidget(self.load_nifti_button)

        self.load_dicom_button = QPushButton('📂 Load DICOM Series')
        self.load_dicom_button.clicked.connect(self.load_dicom_series)
        load_layout.addWidget(self.load_dicom_button)

        self.load_dicom_file_button = QPushButton('📄 Load Single DICOM')
        self.load_dicom_file_button.clicked.connect(self.load_single_dicom)
        load_layout.addWidget(self.load_dicom_file_button)

        self.load_segmentation_button = QPushButton('🎯 Load Segmentation')
        self.load_segmentation_button.clicked.connect(self.load_segmentation)
        self.load_segmentation_button.setEnabled(False)
        load_layout.addWidget(self.load_segmentation_button)

        # highlight-start
        # New simplified toggle button
        self.toggle_outline_button = QPushButton('🔲 Show Outline')
        self.toggle_outline_button.setCheckable(True)
        self.toggle_outline_button.clicked.connect(self.toggle_segmentation_outline)
        self.toggle_outline_button.setEnabled(False)
        load_layout.addWidget(self.toggle_outline_button)
        # highlight-end

        load_group.setLayout(load_layout)
        self.control_layout.addWidget(load_group)

        # === SEGMENTATION VIEW GROUP (REMOVED) ===
        # highlight-start
        # The entire "Segmentation Options" QGroupBox has been removed.
        # highlight-end

        # === ROI GROUP ===
        roi_group = QGroupBox("📐 Region of Interest")
        roi_layout = QVBoxLayout()

        self.draw_roi_button = QPushButton('✏ Draw ROI')
        self.draw_roi_button.setCheckable(True)
        roi_layout.addWidget(self.draw_roi_button)

        self.clear_roi_button = QPushButton('🗑 Clear ROI')
        self.clear_roi_button.clicked.connect(lambda: self.clear_roi())
        roi_layout.addWidget(self.clear_roi_button)

        self.save_roi_button = QPushButton('💾 Save ROI Volume')
        self.save_roi_button.clicked.connect(self.save_roi_volume)
        roi_layout.addWidget(self.save_roi_button)

        roi_group.setLayout(roi_layout)
        self.control_layout.addWidget(roi_group)

        # === PLAYBACK & CONTROLS ===
        controls_group = QGroupBox("⚙ View Controls")
        controls_layout = QVBoxLayout()

        self.play_pause_button = QPushButton("▶ Play/Pause")
        self.play_pause_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_pause_button)

        controls_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'hot', 'cool'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        controls_layout.addWidget(self.colormap_combo)

        self.reset_button = QPushButton("🔄 Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        controls_layout.addWidget(self.reset_button)

        # Show Oblique View
        self.oblique_button = QPushButton("Show Oblique View", self)
        self.oblique_button.setCheckable(True)
        self.oblique_button.clicked.connect(self.toggle_oblique_view)
        self.control_layout.addWidget(self.oblique_button)

        # Angle sliders for oblique view
        angle_layout = QVBoxLayout()
        angle_layout.addWidget(QLabel("Oblique Angle X (degrees)"))
        self.oblique_angle_x_slider = QSlider(Qt.Horizontal)
        self.oblique_angle_x_slider.setRange(-180, 180)
        self.oblique_angle_x_slider.setValue(30)
        self.oblique_angle_x_slider.valueChanged.connect(lambda _: self.show_oblique_view())
        angle_layout.addWidget(self.oblique_angle_x_slider)

        angle_layout.addWidget(QLabel("Oblique Angle Y (degrees)"))
        self.oblique_angle_y_slider = QSlider(Qt.Horizontal)
        self.oblique_angle_y_slider.setRange(-180, 180)
        self.oblique_angle_y_slider.setValue(45)
        self.oblique_angle_y_slider.valueChanged.connect(lambda _: self.show_oblique_view())
        angle_layout.addWidget(self.oblique_angle_y_slider)
        self.control_layout.addLayout(angle_layout)

        controls_group.setLayout(controls_layout)
        self.control_layout.addWidget(controls_group)

        self.control_layout.addStretch()

        # Status bar at bottom
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready to load scan data...", 3000)
        self.control_layout.addWidget(self.status_bar)

    def create_viewport_panel(self):
        """Create the right panel with viewports"""
        self.viewport_panel = QWidget()
        self.viewport_layout = QVBoxLayout()
        self.viewport_panel.setLayout(self.viewport_layout)

        self.axial_fig, self.axial_ax = plt.subplots(facecolor='#2b2b2b')
        self.coronal_fig, self.coronal_ax = plt.subplots(facecolor='#2b2b2b')
        self.sagittal_fig, self.sagittal_ax = plt.subplots(facecolor='#2b2b2b')

        for fig in [self.axial_fig, self.coronal_fig, self.sagittal_fig]:
            fig.patch.set_facecolor('#2b2b2b')
            fig.tight_layout(pad=0.1)

        self.axial_canvas = FigureCanvas(self.axial_fig)
        self.coronal_canvas = FigureCanvas(self.coronal_fig)
        self.sagittal_canvas = FigureCanvas(self.sagittal_fig)

        for canvas, idx in [(self.axial_canvas, 0), (self.coronal_canvas, 1), (self.sagittal_canvas, 2)]:
            canvas.mpl_connect('scroll_event', lambda event, i=idx: self.wheel_zoom(event, i))
            canvas.mpl_connect('button_press_event', lambda event, i=idx: self.on_press(event, i))
            canvas.mpl_connect('motion_notify_event', lambda event, i=idx: self.on_motion(event, i))
            canvas.mpl_connect('button_release_event', lambda event, i=idx: self.on_release(event, i))

        self.axial_slider = QSlider(Qt.Horizontal)
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider = QSlider(Qt.Horizontal)

        self.axial_slider.valueChanged.connect(self.update_axial_slice)
        self.coronal_slider.valueChanged.connect(self.update_coronal_slice)
        self.sagittal_slider.valueChanged.connect(self.update_sagittal_slice)

        self.grid_layout = QGridLayout()
        self.axial_group = self.create_viewport_group("⬆ Axial View", self.axial_canvas, self.axial_slider)
        self.coronal_group = self.create_viewport_group("➡ Coronal View", self.coronal_canvas, self.coronal_slider)
        self.sagittal_group = self.create_viewport_group("⬅ Sagittal View", self.sagittal_canvas, self.sagittal_slider)

        self.grid_layout.addWidget(self.axial_group, 0, 0)
        self.grid_layout.addWidget(self.sagittal_group, 0, 1)
        self.grid_layout.addWidget(self.coronal_group, 1, 0)

        self.viewport_layout.addLayout(self.grid_layout)

    def create_viewport_group(self, title, canvas, slider):
        """Create a group box containing viewport and slider"""
        group = QGroupBox(title)
        group.setObjectName("viewport_group")
        layout = QVBoxLayout()
        layout.addWidget(canvas)

        slider_container = QWidget()
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(slider)
        slider_container.setLayout(slider_layout)
        slider_container.setMaximumHeight(50)

        layout.addWidget(slider_container)
        group.setLayout(layout)
        return group

    # highlight-start
    # Removed update_segmentation_opacity
    # highlight-end

    # ========================================================================
    # FILE LOADING METHODS (WITH IMPROVED ERROR HANDLING)
    # ========================================================================

    @safe_execute(show_error=True)
    def load_nifti(self, *args):
        """Load NIfTI file with comprehensive error handling"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open NIfTI File", "", "NIfTI files (*.nii *.nii.gz);;All files (*)"
        )
        if not file_path:
            return

        self.logger.info(f"Loading NIfTI file: {file_path}")
        self.status_bar.showMessage("⏳ Loading NIfTI file...", 0)

        self.sitk_image = sitk.ReadImage(file_path)
        self.scan_array = sitk.GetArrayFromImage(self.sitk_image)

        if self.scan_array.size == 0:
            raise ValueError("Loaded array is empty")
        if len(self.scan_array.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape: {self.scan_array.shape}")

        self.current_scan_path = file_path
        self.logger.info(f"Loaded NIfTI data with shape: {self.scan_array.shape}")

        self.initialize_viewers()
        self.load_segmentation_button.setEnabled(True)
        self.status_bar.showMessage(
            f"✓ Loaded: {os.path.basename(file_path)} | Shape: {self.scan_array.shape}", 5000
        )

    @safe_execute(show_error=True)
    def load_dicom_series(self, *args):
        """Load DICOM series with error handling"""
        directory_path = QFileDialog.getExistingDirectory(self, "Select DICOM Series Directory")
        if not directory_path:
            return

        self.logger.info(f"Loading DICOM series from: {directory_path}")
        self.status_bar.showMessage("⏳ Loading DICOM series...", 0)

        reader = sitk.ImageSeriesReader()
        dicom_series = reader.GetGDCMSeriesFileNames(directory_path)

        if not dicom_series:
            raise ValueError("No DICOM series found in selected directory")

        reader.SetFileNames(dicom_series)
        self.sitk_image = reader.Execute()
        self.scan_array = sitk.GetArrayFromImage(self.sitk_image)

        if self.scan_array.size == 0:
            raise ValueError("Loaded DICOM series is empty")

        self.current_scan_path = directory_path
        self.logger.info(f"Loaded DICOM series with shape: {self.scan_array.shape}")

        self.initialize_viewers()
        self.load_segmentation_button.setEnabled(True)
        self.status_bar.showMessage(
            f"✓ Loaded: {len(dicom_series)} DICOM images | Shape: {self.scan_array.shape}", 5000
        )
        # Auto-detect orientation
        self.auto_detect_orientation(dicom_series[0])  # Use first DICOM file

    @safe_execute(show_error=True)
    def load_single_dicom(self, *args):
        """Load single DICOM file with error handling"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM File", "", "DICOM files (*.dcm);;All files (*)"
        )
        if not file_path:
            return

        self.logger.info(f"Loading single DICOM: {file_path}")
        self.status_bar.showMessage("⏳ Loading DICOM file...", 0)

        dicom_data = pydicom.dcmread(file_path)

        if not hasattr(dicom_data, 'pixel_array'):
            raise ValueError("DICOM file does not contain pixel data")

        pixel_array = dicom_data.pixel_array

        if hasattr(dicom_data, 'PhotometricInterpretation') and \
                dicom_data.PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = np.max(pixel_array) - pixel_array

        self.scan_array = pixel_array[np.newaxis, :, :]
        self.sitk_image = sitk.GetImageFromArray(self.scan_array)
        self.current_scan_path = file_path

        self.logger.info(f"Loaded single DICOM with shape: {self.scan_array.shape}")

        self.initialize_viewers()
        self.load_segmentation_button.setEnabled(True)
        self.status_bar.showMessage(
            f"✓ Loaded: {os.path.basename(file_path)} | Shape: {self.scan_array.shape}", 5000
        )
        # Auto-detect orientation
        self.auto_detect_orientation(file_path)

    @safe_execute(show_error=True)
    def load_segmentation(self, *args):
        """Load segmentation masks with automatic resampling if needed"""
        if self.scan_array is None:
            QMessageBox.warning(self, "No Scan", "Please load a scan first.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Segmentation File", "",
            "NIfTI files (*.nii *.nii.gz);;NumPy files (*.npy);;All files (*)"
        )
        if not file_path:
            return

        self.logger.info(f"Loading segmentation from: {file_path}")
        self.status_bar.showMessage("⏳ Loading segmentation...", 0)

        # Load segmentation
        if file_path.endswith('.npy'):
            seg_array = np.load(file_path)
        else:
            seg_image = sitk.ReadImage(file_path)
            seg_array = sitk.GetArrayFromImage(seg_image)

        # Check if shapes match
        if seg_array.shape != self.scan_array.shape:
            self.logger.warning(
                f"Segmentation shape {seg_array.shape} doesn't match scan shape {self.scan_array.shape}"
            )

            # Ask user if they want to resample
            reply = QMessageBox.question(
                self,
                'Shape Mismatch',
                f"Segmentation shape {seg_array.shape} doesn't match scan shape {self.scan_array.shape}.\n\n"
                "Do you want to automatically resample the segmentation to match the scan?\n\n"
                "⚠ This may reduce accuracy of the segmentation boundaries.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                # Calculate zoom factors for each dimension
                zoom_factors = [
                    self.scan_array.shape[0] / seg_array.shape[0],
                    self.scan_array.shape[1] / seg_array.shape[1],
                    self.scan_array.shape[2] / seg_array.shape[2]
                ]

                self.logger.info(f"Resampling segmentation with zoom factors: {zoom_factors}")
                self.status_bar.showMessage("⏳ Resampling segmentation...", 0)

                # Resample using nearest neighbor to preserve label values
                seg_array = zoom(seg_array, zoom_factors, order=0)  # order=0 for nearest neighbor

                self.logger.info(f"Resampled segmentation to shape: {seg_array.shape}")
            else:
                self.status_bar.showMessage("❌ Segmentation loading cancelled", 5000)
                return

        self.segmentation_array = seg_array

        unique_labels = np.unique(self.segmentation_array)
        self.logger.info(f"Loaded segmentation with shape: {self.segmentation_array.shape}")
        self.logger.info(f"Unique labels: {unique_labels}")

        # highlight-start
        # Enable the new outline button
        self.toggle_outline_button.setEnabled(True)
        # highlight-end

        self.status_bar.showMessage(
            f"✓ Loaded segmentation: {os.path.basename(file_path)} | {len(unique_labels)} labels", 5000
        )

    # ========================================================================
    # SEGMENTATION VIEW METHODS
    # ========================================================================

    # highlight-start
    @safe_execute(show_error=True)
    def toggle_segmentation_outline(self, checked):
        """Toggle segmentation outline on/off for all main views"""
        self.outline_enabled = checked

        if checked:
            if self.segmentation_array is None:
                QMessageBox.warning(self, "No Segmentation", "Please load a segmentation first.")
                self.toggle_outline_button.setChecked(False)
                self.outline_enabled = False
                return

            self.toggle_outline_button.setText("🔲 Hide Outline")
            self.status_bar.showMessage("✓ Segmentation outline enabled", 3000)
        else:
            self.toggle_outline_button.setText("🔲 Show Outline")
            self.status_bar.showMessage("✓ Segmentation outline disabled", 3000)

        # Redraw all slices to show/hide the outline
        self.update_all_slices()

    # highlight-end

    # highlight-start
    # Removed toggle_surface_outline
    # Removed toggle_segmentation_view
    # Removed change_segmentation_view
    # Removed show_segmentation_view
    # Removed update_segmentation_slice
    # Removed change_outline_view
    # Removed show_outline_view
    # Removed update_outline_slice
    # highlight-end

    # ========================================================================
    # OTHER METHODS
    # ========================================================================
    def auto_detect_orientation(self, dicom_file_path):
        """Automatically detect orientation when DICOM is loaded."""
        try:
            # Read DICOM file
            ds = pydicom.dcmread(dicom_file_path)
            pixel_array = ds.pixel_array

            # Call the prediction function
            predicted_class, confidence = predict_dicom_image(pixel_array)

            # Arabic mapping for display
            arabic_labels = {
                'axial': 'محوري (Axial / عرضي)',
                'coronal': 'جبهّي / كورونال (Coronal)',
                'sagittal': 'سهمي (Sagittal)',
            }

            arabic_name = arabic_labels.get(predicted_class, predicted_class)

            # Show result in status bar
            self.status_bar.showMessage(
                f"✓ Detected: {predicted_class} - {arabic_name} (confidence: {confidence:.1f}%)",
                10000  # Show for 10 seconds
            )

            # Print to console
            print(f"\n=== Auto Orientation Detection ===")
            print(f"Orientation: {predicted_class}")
            print(f"Arabic: {arabic_name}")
            print(f"Confidence: {confidence:.2f}%")
            print("===================================\n")

            # Optional: Show popup notification for low confidence
            if confidence < 70:
                QMessageBox.information(
                    self,
                    "Orientation Detection",
                    f"⚠️ Low confidence detection:\n\n"
                    f"Orientation: {predicted_class}\n"
                    f"Arabic: {arabic_name}\n"
                    f"Confidence: {confidence:.1f}%\n\n"
                    f"Please verify the orientation manually."
                )

        except Exception as e:
            error_msg = f"Auto-detection failed: {str(e)}"
            print(f"Error in auto_detect_orientation: {e}")
            self.status_bar.showMessage(error_msg, 5000)
            # Don't show popup for auto-detection errors to avoid interrupting workflow

    def detect_orientation_action(self):
        """Manual orientation detection triggered by button."""
        # Make sure a scan is loaded
        if not hasattr(self, 'current_scan_path') or not self.current_scan_path:
            QMessageBox.warning(self, "No Scan Loaded",
                                "Please load a DICOM series or file first before detecting orientation.")
            self.status_bar.showMessage("No scan loaded for orientation detection.", 5000)
            return

        try:
            # Determine the DICOM file to use
            if os.path.isdir(self.current_scan_path):
                # It's a directory (DICOM series), use first file
                reader = sitk.ImageSeriesReader()
                dicom_series = reader.GetGDCMSeriesFileNames(self.current_scan_path)
                if not dicom_series:
                    raise Exception("No DICOM files found in directory")
                dicom_file = dicom_series[0]
            else:
                # It's a single file
                dicom_file = self.current_scan_path

            # Read DICOM and predict
            ds = pydicom.dcmread(dicom_file)
            pixel_array = ds.pixel_array
            predicted_class, confidence = predict_dicom_image(pixel_array)

            # Arabic mapping for display
            arabic_labels = {
                'axial': 'محوري (Axial / عرضي)',
                'coronal': 'جبهّي / كورونال (Coronal)',
                'sagittal': 'سهمي (Sagittal)',
            }

            arabic_name = arabic_labels.get(predicted_class, predicted_class)

            # Show result in a message box
            result_msg = f"Detected Orientation: {predicted_class}\n"
            result_msg += f"Arabic: {arabic_name}\n"
            result_msg += f"Confidence: {confidence:.1f}%"

            QMessageBox.information(self, "Orientation Detection Result", result_msg)

            # Also show in status bar
            self.status_bar.showMessage(
                f"Detected: {predicted_class} - {arabic_name} (confidence: {confidence:.1f}%)",
                10000  # Show for 10 seconds
            )

            print(f"=== Manual Orientation Detection ===")
            print(f"Orientation: {predicted_class}")
            print(f"Arabic: {arabic_name}")
            print(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            error_msg = f"Error detecting orientation: {str(e)}"
            QMessageBox.critical(self, "Detection Error", error_msg)
            self.status_bar.showMessage(error_msg, 5000)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    @safe_execute(show_error=True)
    def initialize_viewers(self):
        if self.scan_array is None:
            return

        self.logger.info("Initializing viewers")
        self.clear_roi(reset_bounds=True)

        self.axial_slider.setMaximum(self.scan_array.shape[0] - 1)
        self.coronal_slider.setMaximum(self.scan_array.shape[1] - 1)
        self.sagittal_slider.setMaximum(self.scan_array.shape[2] - 1)

        self.crosshair_x = self.scan_array.shape[2] // 2
        self.crosshair_y = self.scan_array.shape[1] // 2
        self.crosshair_z = self.scan_array.shape[0] // 2

        self.axial_slider.setValue(self.crosshair_z)
        self.coronal_slider.setValue(self.crosshair_y)
        self.sagittal_slider.setValue(self.crosshair_x)

        self.update_all_slices()
        self.logger.info("Viewers initialized successfully")

    def on_press(self, event, view_index):
        if event.button == 1 and event.inaxes:
            if self.draw_roi_button.isChecked():
                self.drawing_roi = True
                self.roi_start_pos = (event.xdata, event.ydata)
                self.active_roi_view_index = view_index
                self.clear_roi(reset_bounds=False)

                patch = plt.Rectangle(self.roi_start_pos, 0, 0, edgecolor='cyan', facecolor='none', lw=2)
                event.inaxes.add_patch(patch)

                if view_index == 0:
                    self.axial_ax.roi_patch = patch
                elif view_index == 1:
                    self.coronal_ax.roi_patch = patch
                elif view_index == 2:
                    self.sagittal_ax.roi_patch = patch

                event.inaxes.figure.canvas.draw_idle()
            else:
                self.update_crosshairs_on_click(event)
        elif event.button == 3:
            self.adjusting_window = True
            self.last_mouse_pos = (event.x, event.y)
            QApplication.setOverrideCursor(Qt.BlankCursor)

    def on_motion(self, event, view_index):
        if self.drawing_roi and event.inaxes and self.roi_start_pos:
            width = event.xdata - self.roi_start_pos[0]
            height = event.ydata - self.roi_start_pos[1]

            if view_index == 0:
                patch = getattr(self.axial_ax, 'roi_patch', None)
            elif view_index == 1:
                patch = getattr(self.coronal_ax, 'roi_patch', None)
            elif view_index == 2:
                patch = getattr(self.sagittal_ax, 'roi_patch', None)

            if patch:
                patch.set_width(width)
                patch.set_height(height)
                event.inaxes.figure.canvas.draw_idle()
        elif self.adjusting_window and self.last_mouse_pos is not None:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]

            self.contrast[view_index] *= (1 + dx / 100.0)
            self.brightness[view_index] += dy

            self.last_mouse_pos = (event.x, event.y)
            self.update_display(view_index)
        elif event.button == 1 and not self.draw_roi_button.isChecked():
            self.update_crosshairs(event)

    def on_release(self, event, view_index):
        if self.drawing_roi and event.button == 1 and event.inaxes:
            self.drawing_roi = False
            self.draw_roi_button.setChecked(False)

            x_start, y_start = self.roi_start_pos
            x_end, y_end = event.xdata, event.ydata

            x_min, x_max = min(x_start, x_end), max(x_start, x_end)
            y_min, y_max = min(y_start, y_end), max(y_start, y_end)

            self.store_roi_bounds(self.active_roi_view_index, x_min, x_max, y_min, y_max)
            self.apply_roi_limits()
            self.update_all_slices()
        elif event.button == 3:
            self.adjusting_window = False
            self.last_mouse_pos = None
            QApplication.restoreOverrideCursor()

    def update_crosshairs_on_click(self, event):
        if event.inaxes is None or event.button != 1:
            return

        if event.inaxes == self.axial_ax:
            self.crosshair_x = int(event.xdata)
            self.crosshair_y = int(event.ydata)
            self.sagittal_slider.setValue(self.crosshair_x)
            self.coronal_slider.setValue(self.crosshair_y)
        elif event.inaxes == self.coronal_ax:
            self.crosshair_x = int(event.xdata)
            self.crosshair_z = self.scan_array.shape[0] - 1 - int(event.ydata)
            self.sagittal_slider.setValue(self.crosshair_x)
            self.axial_slider.setValue(self.crosshair_z)
        elif event.inaxes == self.sagittal_ax:
            self.crosshair_y = int(event.xdata)
            self.crosshair_z = self.scan_array.shape[0] - 1 - int(event.ydata)
            self.coronal_slider.setValue(self.crosshair_y)
            self.axial_slider.setValue(self.crosshair_z)

        self.update_all_slices()

    def update_crosshairs(self, event):
        if event.inaxes and event.button == 1:
            if event.inaxes == self.axial_ax:
                self.crosshair_x, self.crosshair_y = int(event.xdata), int(event.ydata)
                self.sagittal_slider.setValue(self.crosshair_x)
                self.coronal_slider.setValue(self.crosshair_y)
            elif event.inaxes == self.coronal_ax:
                self.crosshair_x = int(event.xdata)
                self.crosshair_z = self.scan_array.shape[0] - 1 - int(event.ydata)
                self.sagittal_slider.setValue(self.crosshair_x)
                self.axial_slider.setValue(self.crosshair_z)
            elif event.inaxes == self.sagittal_ax:
                self.crosshair_y = int(event.xdata)
                self.crosshair_z = self.scan_array.shape[0] - 1 - int(event.ydata)
                self.coronal_slider.setValue(self.crosshair_y)
                self.axial_slider.setValue(self.crosshair_z)
            self.update_all_slices()

    def update_axial_slice(self, value):
        self.crosshair_z = value
        if self.scan_array is not None:
            self.show_axial_slice(self.scan_array, value)
        if self.oblique_enabled:
            self.show_oblique_view()

    def update_coronal_slice(self, value):
        self.crosshair_y = value
        if self.scan_array is not None:
            self.show_coronal_slice(self.scan_array, value)
        if self.oblique_enabled:
            self.show_oblique_view()

    def update_sagittal_slice(self, value):
        self.crosshair_x = value
        if self.scan_array is not None:
            self.show_sagittal_slice(self.scan_array, value)
        if self.oblique_enabled:
            self.show_oblique_view()

    def update_all_slices(self):
        self.update_axial_slice(self.crosshair_z)
        self.update_coronal_slice(self.crosshair_y)
        self.update_sagittal_slice(self.crosshair_x)
        # highlight-start
        # Removed logic for updating 4th panel sliders
        # highlight-end

    def show_axial_slice(self, scan, slice_index):
        self.axial_ax.clear()
        slice_data = scan[slice_index, :, :]
        self.display_slice(self.axial_ax, slice_data, f"Axial View (Slice {slice_index})", 0)

        # highlight-start
        # Draw outline if enabled
        if self.outline_enabled and self.segmentation_array is not None:
            seg_slice = self.segmentation_array[slice_index, :, :]
            self.draw_surface_outline(self.axial_ax, seg_slice)
        # highlight-end

        if self.roi_bounds_3d:
            z_min, z_max, y_min, y_max, x_min, x_max = self.roi_bounds_3d
            if z_min <= slice_index <= z_max:
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     edgecolor='cyan', facecolor='none', lw=2)
                self.axial_ax.add_patch(rect)

        self.axial_ax.axvline(self.crosshair_x, color='#00adb5', linestyle='--', linewidth=1, alpha=0.7)
        self.axial_ax.axhline(self.crosshair_y, color='#00adb5', linestyle='--', linewidth=1, alpha=0.7)
        self.axial_canvas.draw()

    def show_coronal_slice(self, scan, slice_index):
        self.coronal_ax.clear()
        slice_data = np.flipud(scan[:, slice_index, :])
        self.display_slice(self.coronal_ax, slice_data, f"Coronal View (Slice {slice_index})", 1)

        # highlight-start
        # Draw outline if enabled
        if self.outline_enabled and self.segmentation_array is not None:
            seg_slice = np.flipud(self.segmentation_array[:, slice_index, :])
            self.draw_surface_outline(self.coronal_ax, seg_slice)
        # highlight-end

        if self.roi_bounds_3d:
            z_min, z_max, y_min, y_max, x_min, x_max = self.roi_bounds_3d
            if y_min <= slice_index <= y_max:
                z_plot_min = scan.shape[0] - 1 - z_max
                height = z_max - z_min
                rect = plt.Rectangle((x_min, z_plot_min), x_max - x_min, height,
                                     edgecolor='cyan', facecolor='none', lw=2)
                self.coronal_ax.add_patch(rect)

        self.coronal_ax.axvline(self.crosshair_x, color='#00adb5', linestyle='--', linewidth=1, alpha=0.7)
        self.coronal_ax.axhline(scan.shape[0] - 1 - self.crosshair_z, color='#00adb5',
                                linestyle='--', linewidth=1, alpha=0.7)
        self.coronal_canvas.draw()

    def show_sagittal_slice(self, scan, slice_index):
        self.sagittal_ax.clear()
        slice_data = np.flipud(scan[:, :, slice_index])
        self.display_slice(self.sagittal_ax, slice_data, f"Sagittal View (Slice {slice_index})", 2)

        # highlight-start
        # Draw outline if enabled
        if self.outline_enabled and self.segmentation_array is not None:
            seg_slice = np.flipud(self.segmentation_array[:, :, slice_index])
            self.draw_surface_outline(self.sagittal_ax, seg_slice)
        # highlight-end

        if self.roi_bounds_3d:
            z_min, z_max, y_min, y_max, x_min, x_max = self.roi_bounds_3d
            if x_min <= slice_index <= x_max:
                z_plot_min = scan.shape[0] - 1 - z_max
                height = z_max - z_min
                rect = plt.Rectangle((y_min, z_plot_min), y_max - y_min, height,
                                     edgecolor='cyan', facecolor='none', lw=2)
                self.sagittal_ax.add_patch(rect)

        self.sagittal_ax.axvline(self.crosshair_y, color='#00adb5', linestyle='--', linewidth=1, alpha=0.7)
        self.sagittal_ax.axhline(scan.shape[0] - 1 - self.crosshair_z, color='#00adb5',
                                 linestyle='--', linewidth=1, alpha=0.7)
        self.sagittal_canvas.draw()

    def draw_surface_outline(self, ax, seg_slice):
        """Draw ONLY the outer surface outline for segmentation on a given axis"""

        if seg_slice is None or seg_slice.size == 0:
            return

        # Create a binary mask of ALL segmentation (any non-zero value)
        combined_mask = (seg_slice > 0).astype(np.uint8)

        if np.sum(combined_mask) == 0:
            return  # Nothing to draw

        try:
            # --- NEW ROBUST STRATEGY ---
            # 1. Find ALL contours in the combined mask
            contours = measure.find_contours(combined_mask, 0.5)

            if not contours:
                return  # No contours found

            # 2. Find the contour that encloses the largest area
            largest_contour = None
            max_area = -1

            for contour in contours:
                # Get bounding box (min_row, min_col, max_row, max_col)
                min_r, min_c = np.min(contour, axis=0)
                max_r, max_c = np.max(contour, axis=0)
                # Calculate area of the bounding box
                area = (max_r - min_r) * (max_c - min_c)

                if area > max_area:
                    max_area = area
                    largest_contour = contour

            # 3. Draw only the largest contour (which is the outer one)
            if largest_contour is not None:
                ax.plot(largest_contour[:, 1], largest_contour[:, 0],
                        color='#FF3333', linewidth=1.5, alpha=1.0)
            # --- END NEW STRATEGY ---

        except Exception as e:
            # Log the error, which might explain why nothing appeared
            self.logger.error(f"Error drawing outer surface outline: {e}", exc_info=True)

    def display_slice(self, ax, slice_data, title, idx):
        adjusted_slice = (slice_data + self.brightness[idx]) * self.contrast[idx]
        ax.imshow(adjusted_slice, cmap=self.current_colormap,
                  vmin=np.min(slice_data), vmax=np.max(slice_data))
        ax.set_title(title, color='#00adb5', fontsize=11, pad=10, weight='bold')
        ax.axis('off')

    def update_display(self, idx):
        if idx == 0:
            self.update_axial_slice(self.axial_slider.value())
        elif idx == 1:
            self.update_coronal_slice(self.coronal_slider.value())
        elif idx == 2:
            self.update_sagittal_slice(self.sagittal_slider.value())

    def toggle_oblique_view(self, checked):
        self.oblique_enabled = checked
        if checked:
            if hasattr(self, 'oblique_group'):
                self.oblique_group.show()
            self.show_oblique_view()
            self.oblique_button.setText("Hide Oblique View")
        else:
            if hasattr(self, 'oblique_group'):
                self.oblique_group.hide()
            self.oblique_button.setText("Show Oblique View")

    def update_oblique_rotation(self):
        """Update oblique view when rotation sliders change"""
        self.oblique_x_label.setText(f"{self.oblique_x_slider.value()}°")
        self.oblique_y_label.setText(f"{self.oblique_y_slider.value()}°")

        if self.oblique_enabled:
            self.show_oblique_view()

    def show_oblique_view(self):
        if not hasattr(self, 'sitk_image') or self.sitk_image is None:
            return

        try:
            angle_x = self.oblique_angle_x_slider.value()
            angle_y = self.oblique_angle_y_slider.value()

            ax = np.deg2rad(angle_x)
            ay = np.deg2rad(angle_y)

            cross_index = [self.crosshair_x, self.crosshair_y, self.crosshair_z]
            center = self.sitk_image.TransformContinuousIndexToPhysicalPoint(cross_index)

            transform = sitk.Euler3DTransform()
            transform.SetCenter(center)
            transform.SetRotation(ax, ay, 0.0)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(self.sitk_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(transform)
            resampled = resampler.Execute(self.sitk_image)

            arr = sitk.GetArrayFromImage(resampled)
            self.oblique_array = arr

            if not hasattr(self, 'oblique_fig'):
                self.oblique_fig, self.oblique_ax = plt.subplots(facecolor='#1e1e1e')
                self.oblique_fig.patch.set_facecolor('#1e1e1e')
                self.oblique_fig.tight_layout(pad=0.1)
                self.oblique_canvas = FigureCanvas(self.oblique_fig)
                self.oblique_slider = QSlider(Qt.Horizontal)
                self.oblique_slider.setMinimum(0)
                self.oblique_slider.setMaximum(arr.shape[0] - 1)
                self.oblique_slider.setValue(arr.shape[0] // 2)
                self.oblique_slider.valueChanged.connect(self.update_oblique_slice)
                self.oblique_group = self.create_viewport_group("Oblique View", self.oblique_canvas,
                                                                self.oblique_slider)
                self.grid_layout.addWidget(self.oblique_group, 1, 1)

            slice_index = self.oblique_slider.value()
            slice_data = self.oblique_array[slice_index, :, :]
            self.oblique_ax.clear()
            # Oblique view uses the first view's B/C settings for simplicity
            self.display_slice(self.oblique_ax, slice_data, f"Oblique View ({angle_x}°, {angle_y}°)", 0)
            self.oblique_canvas.draw()

        except Exception as e:
            self.status_bar.showMessage(f"Error generating oblique view: {e}")

    def update_oblique_slice(self, value):
        if hasattr(self, 'oblique_array'):
            slice_data = self.oblique_array[value, :, :]
            self.oblique_ax.clear()
            angle_x = self.oblique_angle_x_slider.value()
            angle_y = self.oblique_angle_y_slider.value()
            # Oblique view uses the first view's B/C settings
            self.display_slice(self.oblique_ax, slice_data, f"Oblique View ({angle_x}°, {angle_y}°)", 0)
            self.oblique_canvas.draw()

    def store_roi_bounds(self, view_index, x_min_plot, x_max_plot, y_min_plot, y_max_plot):
        z_s, y_s, x_s = self.scan_array.shape

        if view_index == 0:  # Axial
            x_min, x_max = int(x_min_plot), int(x_max_plot)
            y_min, y_max = int(y_min_plot), int(y_max_plot)
            z_min, z_max = 0, z_s - 1
        elif view_index == 1:  # Coronal
            x_min, x_max = int(x_min_plot), int(x_max_plot)
            z_min, z_max = z_s - 1 - int(y_max_plot), z_s - 1 - int(y_min_plot)
            y_min, y_max = 0, y_s - 1
        elif view_index == 2:  # Sagittal
            y_min, y_max = int(x_min_plot), int(x_max_plot)
            z_min, z_max = z_s - 1 - int(y_max_plot), z_s - 1 - int(y_min_plot)
            x_min, x_max = 0, x_s - 1

        self.roi_bounds_3d = [
            max(0, z_min), min(z_s - 1, z_max),
            max(0, y_min), min(y_s - 1, y_max),
            max(0, x_min), min(x_s - 1, x_max)
        ]

        self.logger.info(f"ROI bounds set: {self.roi_bounds_3d}")

    def apply_roi_limits(self):
        if not self.roi_bounds_3d:
            return

        z_min, z_max, y_min, y_max, x_min, x_max = self.roi_bounds_3d

        self.axial_slider.setMinimum(z_min)
        self.axial_slider.setMaximum(z_max)
        self.coronal_slider.setMinimum(y_min)
        self.coronal_slider.setMaximum(y_max)
        self.sagittal_slider.setMinimum(x_min)
        self.sagittal_slider.setMaximum(x_max)

    def clear_roi(self, reset_bounds=True, *args):
        if reset_bounds:
            self.roi_bounds_3d = None
            if self.scan_array is not None:
                self.axial_slider.setMinimum(0)
                self.axial_slider.setMaximum(self.scan_array.shape[0] - 1)
                self.coronal_slider.setMinimum(0)
                self.coronal_slider.setMaximum(self.scan_array.shape[1] - 1)
                self.sagittal_slider.setMinimum(0)
                self.sagittal_slider.setMaximum(self.scan_array.shape[2] - 1)

        for ax in [self.axial_ax, self.coronal_ax, self.sagittal_ax]:
            if hasattr(ax, 'roi_patch'):
                try:
                    ax.roi_patch.remove()
                except ValueError:
                    pass
                del ax.roi_patch

        self.update_all_slices()
        self.status_bar.showMessage("✓ ROI cleared", 3000)

    def save_roi_volume(self, *args):
        if self.roi_bounds_3d is None:
            QMessageBox.warning(self, "No ROI", "Please draw an ROI before saving.")
            return

        z_min, z_max, y_min, y_max, x_min, x_max = self.roi_bounds_3d
        roi_array = self.scan_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

        new_sitk_image = sitk.GetImageFromArray(roi_array)
        new_sitk_image.SetSpacing(self.sitk_image.GetSpacing())
        new_sitk_image.SetDirection(self.sitk_image.GetDirection())

        new_origin_physical = self.sitk_image.TransformIndexToPhysicalPoint(
            (int(x_min), int(y_min), int(z_min))
        )
        new_sitk_image.SetOrigin(new_origin_physical)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Volume", "", "NIfTI files (*.nii.gz)"
        )

        if file_path:
            sitk.WriteImage(new_sitk_image, file_path)
            self.status_bar.showMessage(
                f"✓ ROI saved to {os.path.basename(file_path)}", 5000
            )

    def update_colormap(self, colormap_name):
        self.current_colormap = colormap_name
        self.update_all_slices()
        self.status_bar.showMessage(f"✓ Colormap changed to {colormap_name}", 3000)

    def toggle_playback(self, *args):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.playback_timer.start(100)
            self.play_pause_button.setText("⏸ Pause")
            self.status_bar.showMessage("▶ Playing...", 0)
        else:
            self.playback_timer.stop()
            self.play_pause_button.setText("▶ Play")
            self.status_bar.showMessage("⏸ Paused", 3000)

    def update_slices(self):
        if not self.is_playing:
            return

        # Advance Axial Slider
        current_axial = self.axial_slider.value()
        if current_axial < self.axial_slider.maximum():
            self.axial_slider.setValue(current_axial + 1)
        else:
            self.axial_slider.setValue(self.axial_slider.minimum())

        # Advance Coronal Slider
        current_coronal = self.coronal_slider.value()
        if current_coronal < self.coronal_slider.maximum():
            self.coronal_slider.setValue(current_coronal + 1)
        else:
            self.coronal_slider.setValue(self.coronal_slider.minimum())

        # Advance Sagittal Slider
        current_sagittal = self.sagittal_slider.value()
        if current_sagittal < self.sagittal_slider.maximum():
            self.sagittal_slider.setValue(current_sagittal + 1)
        else:
            self.sagittal_slider.setValue(self.sagittal_slider.minimum())

    def reset_view(self, *args):
        if self.scan_array is not None:
            self.clear_roi(reset_bounds=True)
            self.brightness = [0, 0, 0]
            self.contrast = [1.0, 1.0, 1.0]
            self.colormap_combo.setCurrentText('gray')
            self.initialize_viewers()
            self.status_bar.showMessage("✓ View reset", 3000)

    def wheel_zoom(self, event, view_index):
        if event.inaxes is None:
            return

        ax = event.inaxes
        scale = 1.1 if event.button == 'up' else 1 / 1.1

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_data, y_data = event.xdata, event.ydata
        if x_data is None:
            return

        new_x_min = x_data - (x_data - x_min) / scale
        new_x_max = x_data + (x_max - x_data) / scale
        new_y_min = y_data - (y_data - y_min) / scale
        new_y_max = y_data + (y_max - y_data) / scale

        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)
        ax.figure.canvas.draw_idle()

    def keyPressEvent(self, event):
        step = 10
        if event.key() == Qt.Key_Left:
            self.pan_view(-step, 0)
        elif event.key() == Qt.Key_Right:
            self.pan_view(step, 0)
        elif event.key() == Qt.Key_Up:
            self.pan_view(0, -step)
        elif event.key() == Qt.Key_Down:
            self.pan_view(0, step)

    def pan_view(self, dx, dy):
        active_canvas = QApplication.instance().widgetAt(QCursor.pos())

        if active_canvas == self.axial_canvas:
            self.pan_specific_view(self.axial_ax, dx, dy)
        elif active_canvas == self.coronal_canvas:
            self.pan_specific_view(self.coronal_ax, dx, dy)
        elif active_canvas == self.sagittal_canvas:
            self.pan_specific_view(self.sagittal_ax, dx, dy)

    def pan_specific_view(self, ax, dx, dy):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.set_xlim(x_min + dx, x_max + dx)
        ax.set_ylim(y_min + dy, y_max + dy)
        ax.figure.canvas.draw_idle()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load and apply stylesheet
    try:
        with open('../style.qss', 'r') as f:
            app.setStyleSheet(f.read())
    except Exception as e:
        print(f"Warning: Could not load style.qss - {e}")

    viewer = MRIViewer()
    viewer.show()
    sys.exit(app.exec_())
