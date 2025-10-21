import sys
import os
import datetime
import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage

class JPG2DICOMApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JPG → DICOM Converter")
        self.setMinimumSize(600, 400)

        self.image = None            # PIL Image
        self.pixel_array = None      # numpy array
        self.image_path = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Preview area
        self.preview_label = QLabel("No image selected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet("QLabel { background: #222; color: #ddd; border: 1px solid #444 }")
        main_layout.addWidget(self.preview_label, stretch=1)

        # Buttons
        button_layout = QHBoxLayout()

        self.select_btn = QPushButton("Select JPG")
        self.select_btn.clicked.connect(self.select_jpg)
        button_layout.addWidget(self.select_btn)

        self.export_btn = QPushButton("Export as DICOM")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_dicom)
        button_layout.addWidget(self.export_btn)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def select_jpg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select JPG/PNG image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return

        try:
            img = Image.open(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image:\n{e}")
            return

        self.image_path = path
        # Keep original image; create numpy array when exporting
        self.image = img.convert("RGB")  # keep RGB for nicer preview even if converted later to grayscale
        qimg = ImageQt.ImageQt(self.image)
        pix = QPixmap.fromImage(qimg)
        # scale to preview label while keeping aspect ratio
        w = max(200, self.preview_label.width() - 20)
        h = max(150, self.preview_label.height() - 20)
        scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        self.preview_label.setText("")  # remove text
        self.export_btn.setEnabled(True)

    def export_dicom(self):
        if self.image is None:
            QMessageBox.warning(self, "No image", "Please select a JPG first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save DICOM file", os.path.splitext(os.path.basename(self.image_path))[0] + ".dcm", "DICOM Files (*.dcm)")
        if not save_path:
            return

        # Convert PIL image to numpy array
        pil = self.image
        arr = np.array(pil)

        # Determine grayscale or RGB output for DICOM:
        # For simplicity we'll export grayscale (MONOCHROME2) by converting to luminance.
        # If you prefer to keep RGB, the code below can be adapted.
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Convert to grayscale using luminance formula
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
            pixel_array = gray
            samples_per_pixel = 1
            photometric = "MONOCHROME2"
            planar_configuration = None
        else:
            # already single channel
            pixel_array = arr.astype(np.uint8)
            samples_per_pixel = 1
            photometric = "MONOCHROME2"
            planar_configuration = None

        try:
            self._save_numpy_as_dicom(pixel_array, save_path,
                                     samples_per_pixel=samples_per_pixel,
                                     photometric=photometric,
                                     planar_configuration=planar_configuration)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save DICOM:\n{e}")
            return

        QMessageBox.information(self, "Saved", f"Successfully saved DICOM:\n{save_path}")

    def _save_numpy_as_dicom(self, pixel_array: np.ndarray, out_path: str,
                             samples_per_pixel=1, photometric="MONOCHROME2", planar_configuration=None):
        """
        Create a minimal FileDataset and save the numpy pixel array as PixelData.
        This uses anonymous metadata (fast option).
        """

        # File meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # Create FileDataset (preable 128 bytes + 'DICM')
        dt = datetime.datetime.now()
        ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Anonymous/basic metadata
        ds.PatientName = "ANON^PATIENT"
        ds.PatientID = "000000"
        ds.Modality = "OT"  # other
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        ds.StudyDate = dt.strftime("%Y%m%d")
        ds.StudyTime = dt.strftime("%H%M%S")
        ds.ContentDate = dt.strftime("%Y%m%d")
        ds.ContentTime = dt.strftime("%H%M%S")

        # Image specifics
        if pixel_array.ndim == 2:
            rows, cols = pixel_array.shape
            ds.Rows = int(rows)
            ds.Columns = int(cols)
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = photometric
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.PixelData = pixel_array.tobytes()
        elif pixel_array.ndim == 3 and pixel_array.shape[2] == 3:
            rows, cols, _ = pixel_array.shape
            ds.Rows = int(rows)
            ds.Columns = int(cols)
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = "RGB"
            ds.PlanarConfiguration = planar_configuration if planar_configuration is not None else 0
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.PixelData = pixel_array.tobytes()
        else:
            raise ValueError("Unsupported image shape for DICOM conversion.")

        # Encoding flags
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Save file
        ds.save_as(out_path, write_like_original=False)

def main():
    app = QApplication(sys.argv)
    w = JPG2DICOMApp()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
