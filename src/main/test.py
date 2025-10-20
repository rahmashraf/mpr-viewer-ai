import sys
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
    QFileDialog, QSlider, QLabel, QComboBox, QFrame, QGridLayout, QGroupBox,
    QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support


class MedicalImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Medical Image Viewer")
        self.setGeometry(50, 50, 1800, 1000)

        # --- Data and State ---
        self.data = None
        self.vtk_initialized = False
        self.image_min = 0.0
        self.image_max = 1.0

        # --- Timers and Widgets Storage ---
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.advance_slice)

        self.sliders = {}
        self.cmaps = {}
        self.brightness_sliders = {}
        self.contrast_sliders = {}
        self.axes = {}
        self.canvases = {}

        # --- Main Layout ---
        self.main_layout = QHBoxLayout(self)

        # --- Left Control Panel ---
        self.setup_control_panel()

        # --- Right View Panel ---
        self.setup_view_panel()

    def setup_control_panel(self):
        """Creates the entire left-side control panel with a scroll area."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(350)

        control_widget = QWidget()
        self.control_layout = QVBoxLayout(control_widget)
        scroll_area.setWidget(control_widget)

        self.main_layout.addWidget(scroll_area)

        # --- Global Controls ---
        global_box = QGroupBox("Global Controls")
        global_layout = QVBoxLayout()
        self.load_button = QPushButton("Load Image (NIfTI / DICOM)")
        self.load_button.clicked.connect(self.load_image)
        global_layout.addWidget(self.load_button)
        global_box.setLayout(global_layout)
        self.control_layout.addWidget(global_box)

        # --- Playback Controls ---
        playback_box = QGroupBox("Playback Controls (Axial)")
        playback_layout = QGridLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 30)  # FPS
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_playback_speed)
        playback_layout.addWidget(self.play_button, 0, 0)
        playback_layout.addWidget(QLabel("Speed (FPS):"), 0, 1)
        playback_layout.addWidget(self.speed_slider, 0, 2)
        playback_box.setLayout(playback_layout)
        self.control_layout.addWidget(playback_box)

        # --- Per-Plane Controls ---
        for name in ["Axial", "Coronal", "Sagittal"]:
            self.control_layout.addWidget(self._create_plane_controls(name))

        self.control_layout.addStretch()

    def _create_plane_controls(self, name):
        """Helper to create a control box for a single plane."""
        group_box = QGroupBox(f"{name} View Controls")
        layout = QVBoxLayout()

        # Colormap
        layout.addWidget(QLabel("Colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(["bone", "gray", "hot", "jet", "viridis", "plasma"])
        cmap_combo.currentIndexChanged.connect(self.update_views)
        self.cmaps[name.lower()] = cmap_combo
        layout.addWidget(cmap_combo)

        # Brightness
        layout.addWidget(QLabel("Brightness:"))
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setRange(-100, 100)
        brightness_slider.setValue(0)
        brightness_slider.valueChanged.connect(self.update_views)
        self.brightness_sliders[name.lower()] = brightness_slider
        layout.addWidget(brightness_slider)

        # Contrast
        layout.addWidget(QLabel("Contrast:"))
        contrast_slider = QSlider(Qt.Horizontal)
        contrast_slider.setRange(1, 400)  # Represents 0.01 to 4.0
        contrast_slider.setValue(100)
        contrast_slider.valueChanged.connect(self.update_views)
        self.contrast_sliders[name.lower()] = contrast_slider
        layout.addWidget(contrast_slider)

        # Slice Slider
        layout.addWidget(QLabel("Slice:"))
        slice_slider = QSlider(Qt.Horizontal)
        slice_slider.setEnabled(False)
        slice_slider.valueChanged.connect(self.update_views)
        self.sliders[name.lower()] = slice_slider
        layout.addWidget(slice_slider)

        group_box.setLayout(layout)
        return group_box

    def setup_view_panel(self):
        """Creates the right-side 2x2 grid for image and 3D views."""
        view_frame = QFrame()
        self.view_layout = QGridLayout(view_frame)
        self.main_layout.addWidget(view_frame)

        # Create 2D views
        for i, name in enumerate(["Axial", "Coronal", "Sagittal"]):
            canvas = FigureCanvas(Figure(figsize=(5, 5)))
            ax = canvas.figure.add_subplot(111)
            ax.set_facecolor('black')
            ax.tick_params(axis='both', colors='white', labelsize=8)
            canvas.figure.patch.set_facecolor('black')
            self.axes[name.lower()] = ax
            self.canvases[name.lower()] = canvas
            canvas.mpl_connect(f'button_press_event', getattr(self, f'on_{name.lower()}_click'))

        # Create 3D view
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.planes = []

        # Add widgets to grid layout
        self.view_layout.addWidget(self.canvases['axial'], 0, 0)
        self.view_layout.addWidget(self.canvases['coronal'], 0, 1)
        self.view_layout.addWidget(self.canvases['sagittal'], 1, 0)
        self.view_layout.addWidget(self.vtk_widget, 1, 1)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                   "Medical Images (*.nii *.nii.gz *.dcm)")
        if not file_path:
            return
        try:
            if file_path.endswith((".nii", ".nii.gz")):
                img = nib.load(file_path)
                self.data = np.transpose(img.get_fdata(), (2, 1, 0)).astype(np.float32)
            elif file_path.endswith(".dcm"):
                dir_path = os.path.dirname(file_path)
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(dir_path)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                self.data = sitk.GetArrayFromImage(image).astype(np.float32)

            if self.data is not None:
                self.image_min = self.data.min()
                self.image_max = self.data.max()
                self.setup_sliders()
                self.setup_vtk_view()
                self.update_views()
        except Exception as e:
            print(f"Error loading image: {e}")

    def setup_sliders(self):
        shape = self.data.shape  # (Z, Y, X)
        self.sliders['axial'].setMaximum(shape[0] - 1)
        self.sliders['axial'].setValue(shape[0] // 2)
        self.sliders['axial'].setEnabled(True)

        self.sliders['coronal'].setMaximum(shape[1] - 1)
        self.sliders['coronal'].setValue(shape[1] // 2)
        self.sliders['coronal'].setEnabled(True)

        self.sliders['sagittal'].setMaximum(shape[2] - 1)
        self.sliders['sagittal'].setValue(shape[2] // 2)
        self.sliders['sagittal'].setEnabled(True)

    def setup_vtk_view(self):
        if self.vtk_initialized:
            self.renderer.RemoveAllViewProps()

        vtk_data_array = numpy_support.numpy_to_vtk(self.data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        self.image_data = vtk.vtkImageData()
        self.image_data.SetDimensions(self.data.shape[2], self.data.shape[1], self.data.shape[0])
        self.image_data.GetPointData().SetScalars(vtk_data_array)

        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(self.image_data)
        iso_value = (self.image_max + self.image_min) / 4.0
        mc.SetValue(0, iso_value)
        mc.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.5, 0.3)
        self.renderer.AddActor(actor)

        self.planes = []
        orientations = [2, 1, 0]  # Z, Y, X for vtkImagePlaneWidget
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

        for i in range(3):
            plane_widget = vtk.vtkImagePlaneWidget()
            plane_widget.SetInputData(self.image_data)
            plane_widget.SetPlaneOrientation(orientations[i])
            plane_widget.SetSliceIndex(self.data.shape[i] // 2)
            plane_widget.DisplayTextOn()
            plane_widget.SetInteractor(self.vtk_widget)
            plane_widget.GetPlaneProperty().SetColor(colors[i])
            plane_widget.On()
            self.planes.append(plane_widget)

        self.renderer.ResetCamera()
        self.vtk_widget.Initialize()
        self.vtk_initialized = True

    def apply_brightness_contrast(self, slice_data, brightness, contrast):
        """Applies brightness and contrast using windowing."""
        # Map slider values to a practical range
        level = brightness * (self.image_max - self.image_min) / 100.0 + (self.image_max + self.image_min) / 2.0
        width = contrast * (self.image_max - self.image_min) / 100.0

        if width == 0: width = 1.0  # Avoid division by zero

        min_val = level - width / 2.0
        max_val = level + width / 2.0

        # Apply windowing and scale to [0, 1] for display
        adjusted_slice = (slice_data - min_val) / (max_val - min_val)
        adjusted_slice = np.clip(adjusted_slice, 0, 1)
        return adjusted_slice

    def update_views(self):
        if self.data is None: return

        z_slice = self.sliders['axial'].value()
        y_slice = self.sliders['coronal'].value()
        x_slice = self.sliders['sagittal'].value()

        # --- Update 2D Views ---
        views = {
            'axial': {'data': self.data[z_slice, :, :], 'h_line': y_slice, 'v_line': x_slice},
            'coronal': {'data': np.rot90(self.data[:, y_slice, :]), 'h_line': z_slice, 'v_line': x_slice},
            'sagittal': {'data': np.rot90(self.data[:, :, x_slice]), 'h_line': z_slice, 'v_line': y_slice}
        }

        for name, view in views.items():
            ax = self.axes[name]
            canvas = self.canvases[name]

            brightness = self.brightness_sliders[name].value()
            contrast = self.contrast_sliders[name].value()
            cmap = self.cmaps[name].currentText()

            adjusted_data = self.apply_brightness_contrast(view['data'], brightness, contrast)

            ax.clear()
            ax.imshow(adjusted_data, cmap=cmap, origin='lower', aspect='equal')
            ax.axhline(view['h_line'], color='lime', linewidth=0.7, alpha=0.8)
            ax.axvline(view['v_line'], color='lime', linewidth=0.7, alpha=0.8)
            ax.set_title(name.capitalize(), color='white', fontsize=10)
            canvas.draw()

        # --- Update 3D View ---
        if self.vtk_initialized and self.planes:
            self.planes[0].SetSliceIndex(z_slice)
            self.planes[1].SetSliceIndex(y_slice)
            self.planes[2].SetSliceIndex(x_slice)
            self.vtk_widget.Render()

    # --- Playback Methods ---
    def toggle_playback(self, checked):
        if self.data is None:
            self.play_button.setChecked(False)
            return

        if checked:
            self.play_button.setText("Pause")
            self.update_playback_speed()
            self.playback_timer.start()
        else:
            self.play_button.setText("Play")
            self.playback_timer.stop()

    def update_playback_speed(self):
        fps = self.speed_slider.value()
        interval = 1000.0 / fps
        self.playback_timer.setInterval(int(interval))

    def advance_slice(self):
        current_slice = self.sliders['axial'].value()
        max_slice = self.sliders['axial'].maximum()
        next_slice = (current_slice + 1) % (max_slice + 1)
        self.sliders['axial'].setValue(next_slice)

    # --- Click Handlers ---
    def on_axial_click(self, event):
        if event.xdata and event.ydata and self.data is not None:
            self.sliders['sagittal'].setValue(int(round(event.xdata)))
            self.sliders['coronal'].setValue(int(round(event.ydata)))

    def on_coronal_click(self, event):
        if event.xdata and event.ydata and self.data is not None:
            self.sliders['sagittal'].setValue(int(round(event.xdata)))
            self.sliders['axial'].setValue(int(round(event.ydata)))

    def on_sagittal_click(self, event):
        if event.xdata and event.ydata and self.data is not None:
            self.sliders['coronal'].setValue(int(round(event.xdata)))
            self.sliders['axial'].setValue(int(round(event.ydata)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MedicalImageViewer()
    viewer.show()
    sys.exit(app.exec_())