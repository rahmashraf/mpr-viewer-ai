import sys
import SimpleITK as sitk
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QWidget, QFileDialog, \
    QSlider, QStatusBar, QGroupBox, QLabel, QComboBox, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pydicom
import os
import vtk
from matplotlib import cm
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support
from scipy.ndimage import rotate
from vtk.util import numpy_support as vtk_numpy_support
import SimpleITK as sitk
import os


class MRIViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self.slices = [0, 0, 0]
        self.marked_points = [[], [], []]
        self.zoom_level = 1.0

        # Configure matplotlib for dark mode
        plt.style.use('dark_background')
        matplotlib.rcParams['figure.facecolor'] = '#1e1e1e'
        matplotlib.rcParams['axes.facecolor'] = '#1e1e1e'
        matplotlib.rcParams['savefig.facecolor'] = '#1e1e1e'
        matplotlib.rcParams['axes.edgecolor'] = '#3e3e42'
        matplotlib.rcParams['axes.labelcolor'] = '#e0e0e0'
        matplotlib.rcParams['xtick.color'] = '#e0e0e0'
        matplotlib.rcParams['ytick.color'] = '#e0e0e0'
        matplotlib.rcParams['text.color'] = '#e0e0e0'
        matplotlib.rcParams['grid.color'] = '#3e3e42'
        matplotlib.rcParams['figure.edgecolor'] = '#1e1e1e'

        self.data = None
        self.slices = [0, 0, 0]
        # ... rest of your __init__ code

        # Brightness/Contrast state
        self.brightness = [0, 0, 0]
        self.contrast = [1.0, 1.0, 1.0]
        self.adjusting_window = False
        self.last_mouse_pos = None

        self.panning = False
        self.pan_start = None
        self.current_colormap = 'gray'
        self.cine_running = False
        self.oblique_enabled = False
        self.sitk_image = None
        self.scan_array = None

        # --- ROI MODIFICATION START ---
        self.drawing_roi = False
        self.roi_start_pos = None
        self.active_roi_view_index = -1
        self.roi_bounds_3d = None  # Stores [z_min, z_max, y_min, y_max, x_min, x_max]
        # --- ROI MODIFICATION END ---

        self.initUI()

    def initUI(self):
        # Basic UI setup
        self.setWindowTitle("MRI Viewer with Segmentation")
        self.setGeometry(100, 100, 1400, 800)

        # Main layout is horizontal to put controls on left
        self.main_layout = QHBoxLayout()

        # Create left control panel
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        self.control_panel.setMaximumWidth(300)

        # Load buttons
        self.load_nifti_button = QPushButton('Load NIfTI Scan', self)
        self.load_nifti_button.clicked.connect(self.load_nifti)
        self.control_layout.addWidget(self.load_nifti_button)

        self.load_dicom_button = QPushButton('Load DICOM Series', self)
        self.load_dicom_button.clicked.connect(self.load_dicom_series)
        self.control_layout.addWidget(self.load_dicom_button)

        self.load_dicom_file_button = QPushButton('Load Single DICOM', self)
        self.load_dicom_file_button.clicked.connect(self.load_single_dicom)
        self.control_layout.addWidget(self.load_dicom_file_button)

        # --- ROI MODIFICATION START ---
        # Add ROI buttons
        roi_group = QGroupBox("Region of Interest (ROI)")
        roi_layout = QVBoxLayout()

        self.draw_roi_button = QPushButton('Draw ROI', self)
        self.draw_roi_button.setCheckable(True)  # Toggle button
        roi_layout.addWidget(self.draw_roi_button)

        self.clear_roi_button = QPushButton('Clear ROI', self)
        self.clear_roi_button.clicked.connect(self.clear_roi)
        roi_layout.addWidget(self.clear_roi_button)

        self.save_roi_button = QPushButton('Save ROI Volume', self)
        self.save_roi_button.clicked.connect(self.save_roi_volume)
        roi_layout.addWidget(self.save_roi_button)

        roi_group.setLayout(roi_layout)
        self.control_layout.addWidget(roi_group)
        # --- ROI MODIFICATION END ---

        # Play/Pause button
        self.play_pause_button = QPushButton("Play/Pause", self)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.control_layout.addWidget(self.play_pause_button)

        # Add Colormap selection dropdown
        colormap_layout = QVBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        colormap_layout.addWidget(self.colormap_combo)
        self.control_layout.addLayout(colormap_layout)

        # Initialize canvases for viewports
        self.axial_fig, self.axial_ax = plt.subplots(facecolor='#1e1e1e')
        self.coronal_fig, self.coronal_ax = plt.subplots(facecolor='#1e1e1e')
        self.sagittal_fig, self.sagittal_ax = plt.subplots(facecolor='#1e1e1e')

        # Set dark background for figures
        self.axial_fig.patch.set_facecolor('#1e1e1e')
        self.coronal_fig.patch.set_facecolor('#1e1e1e')
        self.sagittal_fig.patch.set_facecolor('#1e1e1e')

        # Remove padding/margins for cleaner look
        self.axial_fig.tight_layout(pad=0.1)
        self.coronal_fig.tight_layout(pad=0.1)
        self.sagittal_fig.tight_layout(pad=0.1)

        # Reset button
        self.reset_button = QPushButton("Reset View", self)
        self.reset_button.clicked.connect(self.reset_view)
        self.control_layout.addWidget(self.reset_button)

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

        # Add stretch to push controls to top
        self.control_layout.addStretch()

        # Status bar at bottom of control panel
        self.status_bar = QStatusBar()
        self.control_layout.addWidget(self.status_bar)

        # Add control panel to main layout
        self.main_layout.addWidget(self.control_panel)

        # Create right panel for viewports
        self.viewport_panel = QWidget()
        self.viewport_layout = QVBoxLayout()
        self.viewport_panel.setLayout(self.viewport_layout)

        # Initialize canvases for viewports
        self.axial_fig, self.axial_ax = plt.subplots()
        self.coronal_fig, self.coronal_ax = plt.subplots()
        self.sagittal_fig, self.sagittal_ax = plt.subplots()

        # Create canvas for each figure
        self.axial_canvas = FigureCanvas(self.axial_fig)
        self.coronal_canvas = FigureCanvas(self.coronal_fig)
        self.sagittal_canvas = FigureCanvas(self.sagittal_fig)

        # Connect zoom events
        self.axial_canvas.mpl_connect('scroll_event', lambda event: self.wheel_zoom(event, 0))
        self.coronal_canvas.mpl_connect('scroll_event', lambda event: self.wheel_zoom(event, 1))
        self.sagittal_canvas.mpl_connect('scroll_event', lambda event: self.wheel_zoom(event, 2))

        # --- MODIFICATION: Connect MERGED mouse events ---
        self.axial_canvas.mpl_connect('button_press_event', lambda event: self.on_press(event, 0))
        self.axial_canvas.mpl_connect('motion_notify_event', lambda event: self.on_motion(event, 0))
        self.axial_canvas.mpl_connect('button_release_event', lambda event: self.on_release(event, 0))

        self.coronal_canvas.mpl_connect('button_press_event', lambda event: self.on_press(event, 1))
        self.coronal_canvas.mpl_connect('motion_notify_event', lambda event: self.on_motion(event, 1))
        self.coronal_canvas.mpl_connect('button_release_event', lambda event: self.on_release(event, 1))

        self.sagittal_canvas.mpl_connect('button_press_event', lambda event: self.on_press(event, 2))
        self.sagittal_canvas.mpl_connect('motion_notify_event', lambda event: self.on_motion(event, 2))
        self.sagittal_canvas.mpl_connect('button_release_event', lambda event: self.on_release(event, 2))
        # --- MODIFICATION END ---


        # Initialize crosshair positions
        self.crosshair_x = 0
        self.crosshair_y = 0
        self.crosshair_z = 0

        # --- MODIFICATION: Removed separate crosshair event connections ---
        # (They are now handled inside on_press and on_motion)
        # --- MODIFICATION END ---

        # Initialize crosshair lines
        self.axial_vline = self.axial_ax.axvline(0, color='r', linestyle='--')
        self.axial_hline = self.axial_ax.axhline(0, color='r', linestyle='--')
        self.coronal_vline = self.coronal_ax.axvline(0, color='r', linestyle='--')
        self.coronal_hline = self.coronal_ax.axhline(0, color='r', linestyle='--')
        self.sagittal_vline = self.sagittal_ax.axvline(0, color='r', linestyle='--')
        self.sagittal_hline = self.sagittal_ax.axhline(0, color='r', linestyle='--')

        # Initialize sliders for each view as horizontal
        self.axial_slider = QSlider(Qt.Horizontal)
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider = QSlider(Qt.Horizontal)

        # Connect sliders to update functions
        self.axial_slider.valueChanged.connect(self.update_axial_slice)
        self.coronal_slider.valueChanged.connect(self.update_coronal_slice)
        self.sagittal_slider.valueChanged.connect(self.update_sagittal_slice)

        # Create a grid layout for the viewports
        self.grid_layout = QGridLayout()

        # Create a group for each viewport and slider
        self.axial_group = self.create_viewport_group("Axial View", self.axial_canvas, self.axial_slider)
        self.coronal_group = self.create_viewport_group("Coronal View", self.coronal_canvas, self.coronal_slider)
        self.sagittal_group = self.create_viewport_group("Sagittal View", self.sagittal_canvas, self.sagittal_slider)

        # Add groups to the grid layout
        self.grid_layout.addWidget(self.axial_group, 0, 0)
        self.grid_layout.addWidget(self.sagittal_group, 0, 1)
        self.grid_layout.addWidget(self.coronal_group, 1, 0)

        # Create VTK widget for surface volume
        self.vtk_widget = QVTKRenderWindowInteractor(self.viewport_panel)
        self.vtk_layout = QVBoxLayout()
        self.vtk_layout.addWidget(self.vtk_widget)
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.vtk_interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.vtk_widget.setVisible(False)  # hide initially
        self.viewport_layout.addLayout(self.vtk_layout)
        self.viewport_layout.addLayout(self.grid_layout)

        # Add viewport panel to main layout
        self.main_layout.addWidget(self.viewport_panel)

        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_slices)
        self.is_playing = False

        self.setLayout(self.main_layout)
        # Set focus policy to handle key events
        self.setFocusPolicy(Qt.StrongFocus)

        self.detect_orientation_button = QPushButton("Detect Orientation", self)
        self.detect_orientation_button.clicked.connect(self.detect_orientation_action)
        self.control_layout.addWidget(self.detect_orientation_button)

    # --- MODIFICATION: MERGED MOUSE HANDLER METHODS ---
    def on_press(self, event, view_index):
        if event.button == 1 and event.inaxes:  # Left mouse button
            if self.draw_roi_button.isChecked():
                # Start drawing ROI
                self.drawing_roi = True
                self.roi_start_pos = (event.xdata, event.ydata)
                self.active_roi_view_index = view_index

                # Clear any existing ROI patches
                self.clear_roi(reset_bounds=False)  # Keep bounds, just clear patches

                # Create a new patch
                patch = plt.Rectangle(self.roi_start_pos, 0, 0, edgecolor='cyan', facecolor='none', lw=2)
                event.inaxes.add_patch(patch)

                # Store reference to patch
                if view_index == 0:
                    self.axial_ax.roi_patch = patch
                elif view_index == 1:
                    self.coronal_ax.roi_patch = patch
                elif view_index == 2:
                    self.sagittal_ax.roi_patch = patch

                event.inaxes.figure.canvas.draw_idle()
            else:
                # It's a crosshair click
                self.update_crosshairs_on_click(event)

        elif event.button == 3:  # Right mouse button (Brightness/Contrast)
            self.adjusting_window = True
            self.last_mouse_pos = (event.x, event.y)
            QApplication.setOverrideCursor(Qt.BlankCursor)

    def on_motion(self, event, view_index):
        if self.drawing_roi and event.inaxes and self.roi_start_pos:
            # Update ROI rectangle size
            width = event.xdata - self.roi_start_pos[0]
            height = event.ydata - self.roi_start_pos[1]

            patch = None
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
            # Adjust contrast/brightness
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            self.contrast[view_index] *= (1 + dx / 100.0)
            self.brightness[view_index] += dy
            self.contrast[view_index] = max(0.1, self.contrast[view_index])
            self.last_mouse_pos = (event.x, event.y)
            self.update_display(view_index)

        elif event.button == 1 and not self.draw_roi_button.isChecked():
            # Handle crosshair drag
            self.update_crosshairs(event)

    def on_release(self, event, view_index):
        if self.drawing_roi and event.button == 1 and event.inaxes:
            # Finalize ROI
            self.drawing_roi = False
            self.draw_roi_button.setChecked(False)

            x_start, y_start = self.roi_start_pos
            x_end, y_end = event.xdata, event.ydata

            # Ensure min/max are correct
            x_min, x_max = min(x_start, x_end), max(x_start, x_end)
            y_min, y_max = min(y_start, y_end), max(y_start, y_end)

            # Store 3D bounds and apply limits
            self.store_roi_bounds(self.active_roi_view_index, x_min, x_max, y_min, y_max)
            self.apply_roi_limits()
            self.update_all_slices()  # Redraw all slices with new ROI

        elif event.button == 3:
            # Finalize brightness/contrast
            self.adjusting_window = False
            self.last_mouse_pos = None
            QApplication.restoreOverrideCursor()

    # --- MODIFICATION END ---

    # --- ROI MODIFICATION START: NEW HELPER METHODS ---

    def store_roi_bounds(self, view_index, x_min_plot, x_max_plot, y_min_plot, y_max_plot):
        """Converts 2D plot coordinates into 3D volume indices."""
        if self.scan_array is None:
            return

        (z_shape, y_shape, x_shape) = self.scan_array.shape

        # Clamp plot coordinates to integers within plot limits
        x_min_plot, x_max_plot = int(max(0, x_min_plot)), int(x_max_plot)
        y_min_plot, y_max_plot = int(max(0, y_min_plot)), int(y_max_plot)

        if view_index == 0:  # Axial View (Plot X=vol X, Plot Y=vol Y)
            x_min, x_max = x_min_plot, x_max_plot
            y_min, y_max = y_min_plot, y_max_plot
            z_min, z_max = 0, z_shape - 1  # Full depth

        elif view_index == 1:  # Coronal View (Plot X=vol X, Plot Y=vol Z (flipped))
            x_min, x_max = x_min_plot, x_max_plot
            # Un-flip Z coordinates
            z_max_vol = z_shape - 1 - y_min_plot
            z_min_vol = z_shape - 1 - y_max_plot
            z_min, z_max = z_min_vol, z_max_vol
            y_min, y_max = 0, y_shape - 1  # Full depth

        elif view_index == 2:  # Sagittal View (Plot X=vol Y, Plot Y=vol Z (flipped))
            y_min, y_max = x_min_plot, x_max_plot
            # Un-flip Z coordinates
            z_max_vol = z_shape - 1 - y_min_plot
            z_min_vol = z_shape - 1 - y_max_plot
            z_min, z_max = z_min_vol, z_max_vol
            x_min, x_max = 0, x_shape - 1  # Full depth

        else:
            return

        # Clamp final 3D coordinates to volume shape
        z_min, z_max = max(0, z_min), min(z_shape - 1, z_max)
        y_min, y_max = max(0, y_min), min(y_shape - 1, y_max)
        x_min, x_max = max(0, x_min), min(x_shape - 1, x_max)

        self.roi_bounds_3d = [z_min, z_max, y_min, y_max, x_min, x_max]
        print(f"Set 3D ROI bounds: {self.roi_bounds_3d}")

    def apply_roi_limits(self):
        """Updates slider ranges based on the 3D ROI."""
        if self.roi_bounds_3d is None:
            return

        [z_min, z_max, y_min, y_max, x_min, x_max] = self.roi_bounds_3d

        self.axial_slider.setMinimum(z_min)
        self.axial_slider.setMaximum(z_max)

        self.coronal_slider.setMinimum(y_min)
        self.coronal_slider.setMaximum(y_max)

        self.sagittal_slider.setMinimum(x_min)
        self.sagittal_slider.setMaximum(x_max)

    def clear_roi(self, reset_bounds=True):
        """Clears ROI bounds, resets sliders, and removes drawn patches."""
        if reset_bounds:
            self.roi_bounds_3d = None

            if self.scan_array is not None:
                # Reset sliders to full range
                self.axial_slider.setMinimum(0)
                self.axial_slider.setMaximum(self.scan_array.shape[0] - 1)
                self.coronal_slider.setMinimum(0)
                self.coronal_slider.setMaximum(self.scan_array.shape[1] - 1)
                self.sagittal_slider.setMinimum(0)
                self.sagittal_slider.setMaximum(self.scan_array.shape[2] - 1)

        # Remove any drawn patches
        for ax in [self.axial_ax, self.coronal_ax, self.sagittal_ax]:
            if hasattr(ax, 'roi_patch'):
                try:
                    ax.roi_patch.remove()
                except ValueError:
                    pass  # Already removed
                del ax.roi_patch

        if reset_bounds:
            self.update_all_slices()  # Redraw without ROI boxes
        else:
            # Just redraw to clear patches
            self.axial_canvas.draw_idle()
            self.coronal_canvas.draw_idle()
            self.sagittal_canvas.draw_idle()

    def save_roi_volume(self):
        """Saves the part of the volume defined by the 3D ROI."""
        if self.roi_bounds_3d is None:
            QMessageBox.warning(self, "No ROI", "Please draw an ROI before saving.")
            return

        if self.sitk_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        [z_min, z_max, y_min, y_max, x_min, x_max] = self.roi_bounds_3d

        # Slicing is [z, y, x]. +1 to include the max index.
        limited_array = self.scan_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

        # Create new SITK image from the numpy array
        new_sitk_image = sitk.GetImageFromArray(limited_array)

        # Copy metadata (spacing, direction)
        new_sitk_image.SetSpacing(self.sitk_image.GetSpacing())
        new_sitk_image.SetDirection(self.sitk_image.GetDirection())

        # Calculate the new origin
        # Origin is the physical coordinate of the (0,0,0) index
        # New origin is the physical coordinate of the (x_min, y_min, z_min) index
        old_origin = self.sitk_image.GetOrigin()
        old_spacing = self.sitk_image.GetSpacing()
        old_direction = self.sitk_image.GetDirection()

        # Calculate new origin in physical space
        # Note: SITK index order is (x, y, z)
        new_origin_index = (int(x_min), int(y_min), int(z_min))
        new_origin_physical = self.sitk_image.TransformIndexToPhysicalPoint(new_origin_index)

        new_sitk_image.SetOrigin(new_origin_physical)

        # Open save file dialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save ROI Volume", "",
                                                   "NIfTI files (*.nii.gz);;All files (*)")

        if file_path:
            try:
                sitk.WriteImage(new_sitk_image, file_path)
                self.status_bar.showMessage(f"ROI volume saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ROI volume: {str(e)}")

    # --- ROI MODIFICATION END ---

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

    def create_viewport_group(self, title, canvas, slider):
        """Create a group box containing the viewport and horizontal slider."""
        group = QGroupBox(title)
        group.setObjectName("viewport_group")  # Add this line
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

    def update_crosshairs_on_click(self, event):
        # This is now called from on_press()
        if event.inaxes is None or self.scan_array is None or event.button != 1:  # Only on left-click
            return

        if event.inaxes == self.axial_ax:
            xlim, ylim = self.axial_ax.get_xlim(), self.axial_ax.get_ylim()
        elif event.inaxes == self.coronal_ax:
            xlim, ylim = self.coronal_ax.get_xlim(), self.coronal_ax.get_ylim()
        elif event.inaxes == self.sagittal_ax:
            xlim, ylim = self.sagittal_ax.get_xlim(), self.sagittal_ax.get_ylim()
        else:
            return

        if event.inaxes == self.axial_ax:
            self.crosshair_x = int(event.xdata)
            self.crosshair_y = int(event.ydata)
            self.axial_slider.setValue(self.crosshair_z)
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

        if event.inaxes == self.axial_ax:
            self.axial_ax.set_xlim(xlim)
            self.axial_ax.set_ylim(ylim)
            self.axial_canvas.draw_idle()
        elif event.inaxes == self.coronal_ax:
            self.coronal_ax.set_xlim(xlim)
            self.coronal_ax.set_ylim(ylim)
            self.coronal_canvas.draw_idle()
        elif event.inaxes == self.sagittal_ax:
            self.sagittal_ax.set_xlim(xlim)
            self.sagittal_ax.set_ylim(ylim)
            self.sagittal_canvas.draw_idle()

    def load_nifti(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "",
                                                   "NIfTI files (*.nii *.nii.gz);;All files (*)")
        if file_path:
            try:
                self.sitk_image = sitk.ReadImage(file_path)
                self.scan_array = sitk.GetArrayFromImage(self.sitk_image)
                print(f"Loaded NIfTI data with shape: {self.scan_array.shape}")
                self.initialize_viewers()
                self.status_bar.showMessage(f"Loaded NIfTI: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load NIfTI file: {str(e)}")

    def load_dicom_series(self):
        directory_path = QFileDialog.getExistingDirectory(self, "Select DICOM Series Directory")
        if directory_path:
            try:
                reader = sitk.ImageSeriesReader()
                dicom_series = reader.GetGDCMSeriesFileNames(directory_path)

                if not dicom_series:
                    QMessageBox.warning(self, "Warning", "No DICOM series found in selected directory")
                    return

                reader.SetFileNames(dicom_series)
                self.sitk_image = reader.Execute()
                self.scan_array = sitk.GetArrayFromImage(self.sitk_image)
                print(f"Loaded DICOM series with shape: {self.scan_array.shape}")
                self.initialize_viewers()
                self.status_bar.showMessage(f"Loaded DICOM series: {len(dicom_series)} images")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM series: {str(e)}")

    def load_single_dicom(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "",
                                                   "DICOM files (*.dcm);;All files (*)")
        if file_path:
            try:
                dicom_data = pydicom.dcmread(file_path)
                if hasattr(dicom_data, 'pixel_array'):
                    pixel_array = dicom_data.pixel_array

                    if hasattr(dicom_data, 'PhotometricInterpretation'):
                        if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                            pixel_array = np.max(pixel_array) - pixel_array

                    self.scan_array = pixel_array[np.newaxis, :, :]

                    self.sitk_image = sitk.GetImageFromArray(self.scan_array)

                    print(f"Loaded single DICOM with shape: {self.scan_array.shape}")
                    self.initialize_viewers()
                    self.status_bar.showMessage(f"Loaded DICOM: {os.path.basename(file_path)}")
                else:
                    QMessageBox.warning(self, "Warning", "DICOM file does not contain pixel data")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM file: {str(e)}")

    def initialize_viewers(self):
        if self.scan_array is None:
            return

        # Clear any old ROIs
        self.clear_roi(reset_bounds=True)

        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(self.scan_array.shape[0] - 1)
        self.coronal_slider.setMinimum(0)
        self.coronal_slider.setMaximum(self.scan_array.shape[1] - 1)
        self.sagittal_slider.setMinimum(0)
        self.sagittal_slider.setMaximum(self.scan_array.shape[2] - 1)

        self.crosshair_x = self.scan_array.shape[2] // 2
        self.crosshair_y = self.scan_array.shape[1] // 2
        self.crosshair_z = self.scan_array.shape[0] // 2

        self.axial_slider.setValue(self.crosshair_z)
        self.coronal_slider.setValue(self.crosshair_y)
        self.sagittal_slider.setValue(self.crosshair_x)

        self.update_all_slices()

    def update_crosshairs(self, event):
        # This is now called from on_motion()
        if event.inaxes and event.button == 1:  # Left click and drag
            if event.inaxes == self.axial_ax:
                self.crosshair_x = int(event.xdata)
                self.crosshair_y = int(event.ydata)
                self.sagittal_slider.setValue(int(self.crosshair_x))
                self.coronal_slider.setValue(int(self.crosshair_y))
            elif event.inaxes == self.coronal_ax:
                self.crosshair_x = int(event.xdata)
                self.crosshair_z = self.scan_array.shape[0] - 1 - int(event.ydata)
                self.sagittal_slider.setValue(int(self.crosshair_x))
                self.axial_slider.setValue(int(self.crosshair_z))
            elif event.inaxes == self.sagittal_ax:
                self.crosshair_y = int(event.xdata)
                self.crosshair_z = self.scan_array.shape[0] - 1 - int(event.ydata)
                self.coronal_slider.setValue(int(self.crosshair_y))
                self.axial_slider.setValue(int(self.crosshair_z))

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

    def show_axial_slice(self, scan, slice_index):
        if slice_index >= scan.shape[0]:
            return
        self.axial_ax.clear()
        slice_data = scan[slice_index, :, :]
        self.display_slice(self.axial_ax, slice_data, "Axial View", 0)

        # --- ROI MODIFICATION: Draw ROI Box ---
        if self.roi_bounds_3d:
            [z_min, z_max, y_min, y_max, x_min, x_max] = self.roi_bounds_3d
            # Only draw if this slice is within the Z-bounds
            if z_min <= slice_index <= z_max:
                width = x_max - x_min
                height = y_max - y_min
                rect = plt.Rectangle((x_min, y_min), width, height, edgecolor='cyan', facecolor='none', lw=2)
                self.axial_ax.add_patch(rect)
                self.axial_ax.roi_patch = rect  # Store ref
        # --- ROI MODIFICATION END ---

        self.axial_vline = self.axial_ax.axvline(self.crosshair_x, color='r', linestyle='--')
        self.axial_hline = self.axial_ax.axhline(self.crosshair_y, color='r', linestyle='--')
        self.axial_ax.plot(self.crosshair_x, self.crosshair_y, 'ro', markersize=5)
        self.axial_canvas.draw()

    def show_coronal_slice(self, scan, slice_index):
        if slice_index >= scan.shape[1]:
            return
        self.coronal_ax.clear()
        slice_data = scan[:, slice_index, :]
        slice_data_flipped = np.flipud(slice_data)
        self.display_slice(self.coronal_ax, slice_data_flipped, "Coronal View", 1)

        # --- ROI MODIFICATION: Draw ROI Box ---
        if self.roi_bounds_3d:
            [z_min, z_max, y_min, y_max, x_min, x_max] = self.roi_bounds_3d
            if y_min <= slice_index <= y_max:
                width = x_max - x_min
                # Z-axis is flipped on this plot
                z_plot_min = self.scan_array.shape[0] - 1 - z_max
                z_plot_max = self.scan_array.shape[0] - 1 - z_min
                height = z_plot_max - z_plot_min
                rect = plt.Rectangle((x_min, z_plot_min), width, height, edgecolor='cyan', facecolor='none', lw=2)
                self.coronal_ax.add_patch(rect)
                self.coronal_ax.roi_patch = rect  # Store ref
        # --- ROI MODIFICATION END ---

        self.coronal_vline = self.coronal_ax.axvline(self.crosshair_x, color='r', linestyle='--')
        self.coronal_hline = self.coronal_ax.axhline(self.scan_array.shape[0] - 1 - self.crosshair_z, color='r',
                                                     linestyle='--')
        self.coronal_ax.plot(self.crosshair_x, self.scan_array.shape[0] - 1 - self.crosshair_z, 'ro', markersize=5)
        self.coronal_canvas.draw()

    def show_sagittal_slice(self, scan, slice_index):
        if slice_index >= scan.shape[2]:
            return
        self.sagittal_ax.clear()
        slice_data = scan[:, :, slice_index]
        slice_data_flipped = np.flipud(slice_data)
        self.display_slice(self.sagittal_ax, slice_data_flipped, "Sagittal View", 2)

        # --- ROI MODIFICATION: Draw ROI Box ---
        if self.roi_bounds_3d:
            [z_min, z_max, y_min, y_max, x_min, x_max] = self.roi_bounds_3d
            if x_min <= slice_index <= x_max:
                width = y_max - y_min  # Y-axis is X on this plot
                # Z-axis is flipped
                z_plot_min = self.scan_array.shape[0] - 1 - z_max
                z_plot_max = self.scan_array.shape[0] - 1 - z_min
                height = z_plot_max - z_plot_min
                rect = plt.Rectangle((y_min, z_plot_min), width, height, edgecolor='cyan', facecolor='none', lw=2)
                self.sagittal_ax.add_patch(rect)
                self.sagittal_ax.roi_patch = rect  # Store ref
        # --- ROI MODIFICATION END ---

        self.sagittal_vline = self.sagittal_ax.axvline(self.crosshair_y, color='r', linestyle='--')
        self.sagittal_hline = self.sagittal_ax.axhline(self.scan_array.shape[0] - 1 - self.crosshair_z, color='r',
                                                       linestyle='--')
        self.sagittal_ax.plot(self.crosshair_y, self.scan_array.shape[0] - 1 - self.crosshair_z, 'ro', markersize=5)
        self.sagittal_canvas.draw()

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

    def display_slice(self, ax, slice_data, title, idx):
        if slice_data is None:
            return

        # Use class attributes for brightness/contrast
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)

        # Apply brightness and contrast
        adjusted_slice = (slice_data + self.brightness[idx]) * self.contrast[idx]

        ax.imshow(adjusted_slice, cmap=self.current_colormap, vmin=min_val, vmax=max_val)
        ax.set_title(title, color='#e0e0e0', fontsize=11, pad=10)

        # Remove axes, ticks, and labels for clean look
        ax.axis('off')

        # Set dark background
        ax.set_facecolor('#1e1e1e')
        ax.figure.patch.set_facecolor('#1e1e1e')

    def update_display(self, idx):
        if idx == 0:
            self.update_axial_slice(self.axial_slider.value())
        elif idx == 1:
            self.update_coronal_slice(self.coronal_slider.value())
        elif idx == 2:
            self.update_sagittal_slice(self.sagittal_slider.value())

    def update_oblique_slice(self, value):
        if hasattr(self, 'oblique_array'):
            slice_data = self.oblique_array[value, :, :]
            self.oblique_ax.clear()
            angle_x = self.oblique_angle_x_slider.value()
            angle_y = self.oblique_angle_y_slider.value()
            # Oblique view uses the first view's B/C settings
            self.display_slice(self.oblique_ax, slice_data, f"Oblique View ({angle_x}°, {angle_y}°)", 0)
            self.oblique_canvas.draw()

    def update_colormap(self, colormap_name):
        self.current_colormap = colormap_name
        self.update_all_slices()

    def toggle_playback(self):
        if self.is_playing:
            self.playback_timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.playback_timer.start(100)
            self.play_pause_button.setText("Pause")
        self.is_playing = not self.is_playing

    def update_slices(self):
        if not self.is_playing:
            return

        # Use the slider's min/max for cine loop
        current_axial_value = self.axial_slider.value()
        min_val = self.axial_slider.minimum()
        max_val = self.axial_slider.maximum()

        if current_axial_value < max_val:
            self.axial_slider.setValue(current_axial_value + 1)
        else:
            self.axial_slider.setValue(min_val)  # Loop back to min

    def reset_view(self):
        if self.scan_array is not None:
            # Clear ROI first to reset slider limits
            self.clear_roi(reset_bounds=True)

            self.crosshair_x = self.scan_array.shape[2] // 2
            self.crosshair_y = self.scan_array.shape[1] // 2
            self.crosshair_z = self.scan_array.shape[0] // 2

            self.axial_slider.setValue(self.crosshair_z)
            self.coronal_slider.setValue(self.crosshair_y)
            self.sagittal_slider.setValue(self.crosshair_x)

            # Reset brightness/contrast arrays
            self.brightness = [0, 0, 0]
            self.contrast = [1.0, 1.0, 1.0]

            self.current_colormap = 'gray'
            self.update_all_slices()
            self.status_bar.showMessage("View reset to default")

    def wheel_zoom(self, event, view_index):
        if event.inaxes is None:
            return

        ax = event.inaxes
        base_scale = 1.1

        if event.button == 'up':
            scale_factor = base_scale
        elif event.button == 'down':
            scale_factor = 1 / base_scale
        else:
            return

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_data = event.xdata
        y_data = event.ydata

        if x_data is None or y_data is None:
            return

        x_range = x_max - x_min
        y_range = y_max - y_min
        rel_x = (x_data - x_min) / x_range
        rel_y = (y_data - y_min) / y_range

        new_x_range = x_range / scale_factor
        new_y_range = y_range / scale_factor

        new_x_min = x_data - rel_x * new_x_range
        new_x_max = new_x_min + new_x_range
        new_y_min = y_data - rel_y * new_y_range
        new_y_max = new_y_min + new_y_range

        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)

        if view_index == 0:
            self.axial_canvas.draw_idle()
        elif view_index == 1:
            self.coronal_canvas.draw_idle()
        elif view_index == 2:
            self.sagittal_canvas.draw_idle()

    def keyPressEvent(self, event):
        step_size = 10
        if event.key() == Qt.Key_Left:
            self.pan_view(-step_size, 0)
        elif event.key() == Qt.Key_Right:
            self.pan_view(step_size, 0)
        elif event.key() == Qt.Key_Up:
            self.pan_view(0, -step_size)
        elif event.key() == Qt.Key_Down:
            self.pan_view(0, step_size)

    def pan_view(self, dx, dy):
        mouse_pos = QApplication.instance().widgetAt(QCursor.pos())
        if mouse_pos == self.axial_canvas:
            self.pan_specific_view(self.axial_ax, dx, dy)
        elif mouse_pos == self.coronal_canvas:
            self.pan_specific_view(self.coronal_ax, dx, dy)
        elif mouse_pos == self.sagittal_canvas:
            self.pan_specific_view(self.sagittal_ax, dx, dy)

    def pan_specific_view(self, ax, dx, dy):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        new_xlim = (xlim[0] + dx, xlim[1] + dx)
        new_ylim = (ylim[0] + dy, ylim[1] + dy)
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.figure.canvas.draw_idle()

    def detect_orientation_action(self):
        # Make sure a scan is loaded
        if not hasattr(self, 'current_scan_folder') or not self.current_scan_folder:
            self.status_bar.showMessage("No scan loaded for orientation detection.", 5000)
            return

        try:
            orientation, conf, _ = detect_orientation_from_path(self.current_scan_folder)
            self.status_bar.showMessage(
                f"Detected orientation: {orientation} (confidence: {conf:.2f})", 8000
            )
        except Exception as e:
            self.status_bar.showMessage(f"Error detecting orientation: {str(e)}", 5000)


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