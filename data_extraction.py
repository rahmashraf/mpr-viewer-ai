import nibabel as nib
import numpy as np
import os
from pathlib import Path
import imageio.v2 as imageio  # newer imageio versions need .v2 for imwrite


def normalize_slice(slice_data):
    """Normalize a 2D slice to 0–255 (uint8) for saving as an image."""
    slice_data = np.nan_to_num(slice_data)
    slice_data = slice_data - slice_data.min()
    if slice_data.max() != 0:
        slice_data = slice_data / slice_data.max()
    return (slice_data * 255).astype(np.uint8)


def save_slices(nii_path, output_dir):
    # Load NIfTI volume
    img = nib.load(nii_path)
    data = img.get_fdata()

    # Ensure output folders exist
    output_dir = Path(output_dir)
    (output_dir / "axial").mkdir(parents=True, exist_ok=True)
    (output_dir / "coronal").mkdir(parents=True, exist_ok=True)
    (output_dir / "sagittal").mkdir(parents=True, exist_ok=True)

    # Get file base name (without extension)
    base_name = Path(nii_path).stem.replace(".nii", "")

    print(f"Loaded {nii_path} with shape {data.shape} (X, Y, Z)")

    # Axial view (XY plane, iterate along Z)
    for i in range(data.shape[2]):
        slice_img = normalize_slice(data[:, :, i])
        save_path = output_dir / "axial" / f"{base_name}_axial_{i:03d}.png"
        imageio.imwrite(save_path, np.rot90(slice_img))

    # Coronal view (XZ plane, iterate along Y)
    for i in range(data.shape[1]):
        slice_img = normalize_slice(data[:, i, :])
        save_path = output_dir / "coronal" / f"{base_name}_coronal_{i:03d}.png"
        imageio.imwrite(save_path, np.rot90(slice_img))

    # Sagittal view (YZ plane, iterate along X)
    for i in range(data.shape[0]):
        slice_img = normalize_slice(data[i, :, :])
        save_path = output_dir / "sagittal" / f"{base_name}_sagittal_{i:03d}.png"
        imageio.imwrite(save_path, np.rot90(slice_img))

    print(f"✅ Done! Saved slices to {output_dir}/[axial|coronal|sagittal]")


# Example usage
if __name__ == "__main__":
    nii_file = "sub-0003_T1w.nii.gz"  # path to your NIfTI file
    output_folder = "output_slices"  # where to save slices
    save_slices(nii_file, output_folder)
