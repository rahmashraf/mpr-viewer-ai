# detect_orientation.py
import os
import argparse
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.transform import resize
import matplotlib.pyplot as plt

def read_series_sitk(folder):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(folder)
    if series_IDs:
        files = reader.GetGDCMSeriesFileNames(folder, series_IDs[0])
        reader.SetFileNames(files)
        image = reader.Execute()
        arr = sitk.GetArrayFromImage(image)  # shape: (slices, H, W)
        return arr, files
    # fallback: read .dcm files sorted by filename using pydicom
    files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith('.dcm')])
    if not files:
        raise FileNotFoundError("No .dcm files found in folder: " + folder)
    stacks = []
    for f in files:
        d = pydicom.dcmread(f)
        pa = d.pixel_array.astype(np.float32)
        # apply rescale if present
        slope = float(d.get('RescaleSlope', 1.0))
        intercept = float(d.get('RescaleIntercept', 0.0))
        pa = pa * slope + intercept
        stacks.append(pa)
    return np.stack(stacks, axis=0), files

def detect_from_dicom_file(dcm_path):
    try:
        d = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    except Exception as e:
        return None, 0.0
    if hasattr(d, 'ImageOrientationPatient'):
        iop = [float(x) for x in d.ImageOrientationPatient]  # 6 values
        row = np.array(iop[:3], dtype=float)
        col = np.array(iop[3:], dtype=float)
        normal = np.cross(row, col)
        absn = np.abs(normal)
        idx = int(np.argmax(absn))
        labels = ['sagittal', 'coronal', 'axial']  # idx 0->x,1->y,2->z
        conf = float(absn[idx] / (absn.sum() + 1e-12))
        return labels[idx], conf
    return None, 0.0

def detect_by_volume(arr):
    # arr shape: (Z, H, W)
    # compute per-axis projection variance — axis with biggest variance -> slice-axis
    s_z = np.sum(arr, axis=(1,2))  # length Z
    s_y = np.sum(arr, axis=(0,2))  # length H
    s_x = np.sum(arr, axis=(0,1))  # length W
    vars = [float(np.var(s_z)), float(np.var(s_y)), float(np.var(s_x))]
    idx = int(np.argmax(vars))
    labels = ['axial', 'coronal', 'sagittal']  # index 0->Z,1->H,2->W
    conf = vars[idx] / (sum(vars) + 1e-12)
    return labels[idx], conf

def detect_orientation_from_path(input_path, visualize=False):
    # same as main, but instead of print(), return values
    # returns: final_orientation, final_conf, arr
    if os.path.isdir(input_path):
        arr, files = read_series_sitk(input_path)
        sample_file = files[0] if files else None
    elif os.path.isfile(input_path) and input_path.lower().endswith('.dcm'):
        sample_file = input_path
        d = pydicom.dcmread(input_path)
        pa = d.pixel_array.astype(np.float32)
        slope = float(d.get('RescaleSlope', 1.0))
        intercept = float(d.get('RescaleIntercept', 0.0))
        pa = pa * slope + intercept
        arr = np.expand_dims(pa, axis=0)
    else:
        raise ValueError("Input must be a folder of DICOMs or a .dcm file")

    orientation, conf_meta = (None, 0.0)
    if sample_file:
        orientation, conf_meta = detect_from_dicom_file(sample_file)

    orientation2, conf_vol = detect_by_volume(arr)
    final_orientation = orientation if orientation is not None else orientation2
    final_conf = conf_meta if orientation is not None else conf_vol

    if visualize:
        visualize_middle(arr, title=f"Detected: {final_orientation} (conf {final_conf:.2f})")

    return final_orientation, final_conf, arr


def visualize_middle(arr, title="middle slice"):
    mid = arr.shape[0] // 2
    plt.imshow(arr[mid], cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def main(input_path, visualize=False):
    # decide if folder or single file
    if os.path.isdir(input_path):
        arr, files = read_series_sitk(input_path)
        sample_file = files[0] if files else None
    elif os.path.isfile(input_path) and input_path.lower().endswith('.dcm'):
        # try to read single file first; if single -> try reading folder of that file
        sample_file = input_path
        # try to load pixel_array into a 3D volume of 1 slice
        d = pydicom.dcmread(input_path)
        pa = d.pixel_array.astype(np.float32)
        slope = float(d.get('RescaleSlope', 1.0))
        intercept = float(d.get('RescaleIntercept', 0.0))
        pa = pa * slope + intercept
        arr = np.expand_dims(pa, axis=0)
    else:
        raise ValueError("Input must be a folder of DICOMs or a .dcm file")

    # 1) try metadata-based detection (most reliable)
    orientation, conf_meta = (None, 0.0)
    if sample_file:
        orientation, conf_meta = detect_from_dicom_file(sample_file)

    # 2) fallback to volume-based heuristic if metadata missing
    orientation2, conf_vol = detect_by_volume(arr)

    # decide final
    final_orientation = orientation if orientation is not None else orientation2
    final_conf = conf_meta if orientation is not None else conf_vol

    print("=== Orientation detection result ===")
    print(f"Method used: {'metadata' if orientation is not None else 'volume-heuristic (fallback)'}")
    print(f"Detected orientation: {final_orientation}")
    print(f"Confidence (0-1): {final_conf:.3f}")

    if visualize:
        visualize_middle(arr, title=f"Detected: {final_orientation} (conf {final_conf:.2f})")

    # Save a small npy ready-to-inspect (and optionally resize for MedMNIST)
    out_npy = "detected_volume.npy"
    np.save(out_npy, arr.astype(np.float32))
    print(f"Saved raw volume to: {out_npy} (shape: {arr.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect orientation (axial/coronal/sagittal) for a DICOM folder or file.")
    parser.add_argument("--input", "-i", required=True, help="Path to DICOM folder or single .dcm file")
    parser.add_argument("--visualize", "-v", action="store_true", help="Show middle slice")
    args = parser.parse_args()
    main(args.input, args.visualize)