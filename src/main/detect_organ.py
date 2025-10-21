"""
Organ Detection Module using TotalSegmentator
This module provides organ detection and segmentation for medical images.
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional
import subprocess
import os
import tempfile
import json


class OrganDetector:
    """
    Handles organ detection using TotalSegmentator pre-trained model.
    TotalSegmentator can segment 104 anatomical structures including:
    - Major organs (liver, kidneys, spleen, pancreas, etc.)
    - Bones
    - Muscles
    - Vessels
    """

    # Common organs with their label IDs and colors for visualization
    ORGAN_INFO = {
        'spleen': {'id': 1, 'color': (128, 0, 0), 'name': 'Spleen / الطحال'},
        'kidney_right': {'id': 2, 'color': (0, 128, 0), 'name': 'Right Kidney / الكلية اليمنى'},
        'kidney_left': {'id': 3, 'color': (0, 255, 0), 'name': 'Left Kidney / الكلية اليسرى'},
        'gallbladder': {'id': 4, 'color': (128, 128, 0), 'name': 'Gallbladder / المرارة'},
        'liver': {'id': 5, 'color': (128, 0, 128), 'name': 'Liver / الكبد'},
        'stomach': {'id': 6, 'color': (0, 128, 128), 'name': 'Stomach / المعدة'},
        'pancreas': {'id': 7, 'color': (255, 0, 0), 'name': 'Pancreas / البنكرياس'},
        'adrenal_gland_right': {'id': 8, 'color': (0, 255, 255), 'name': 'Right Adrenal / الغدة الكظرية اليمنى'},
        'adrenal_gland_left': {'id': 9, 'color': (255, 255, 0), 'name': 'Left Adrenal / الغدة الكظرية اليسرى'},
        'lung_upper_lobe_left': {'id': 10, 'color': (255, 128, 0), 'name': 'Left Upper Lung / الرئة العليا اليسرى'},
        'lung_lower_lobe_left': {'id': 11, 'color': (128, 255, 0), 'name': 'Left Lower Lung / الرئة السفلى اليسرى'},
        'lung_upper_lobe_right': {'id': 12, 'color': (0, 128, 255), 'name': 'Right Upper Lung / الرئة العليا اليمنى'},
        'lung_middle_lobe_right': {'id': 13, 'color': (255, 0, 128), 'name': 'Right Middle Lung / الرئة الوسطى اليمنى'},
        'lung_lower_lobe_right': {'id': 14, 'color': (128, 0, 255), 'name': 'Right Lower Lung / الرئة السفلى اليمنى'},
        'heart': {'id': 15, 'color': (255, 0, 255), 'name': 'Heart / القلب'},
        'aorta': {'id': 16, 'color': (192, 192, 0), 'name': 'Aorta / الشريان الأبهر'},
        'bladder': {'id': 17, 'color': (0, 192, 192), 'name': 'Bladder / المثانة'},
    }

    def __init__(self):
        self.segmentation_mask = None
        self.detected_organs = []
        self.organ_bounds = {}

    def check_totalsegmentator_installed(self) -> bool:
        """Check if TotalSegmentator is installed."""
        try:
            result = subprocess.run(['TotalSegmentator', '--version'],
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def install_totalsegmentator(self) -> Tuple[bool, str]:
        """
        Attempt to install TotalSegmentator.
        Returns (success, message)
        """
        try:
            subprocess.run(['pip', 'install', 'TotalSegmentator'],
                           check=True, capture_output=True, timeout=300)
            return True, "TotalSegmentator installed successfully"
        except subprocess.CalledProcessError as e:
            return False, f"Installation failed: {e.stderr.decode()}"
        except Exception as e:
            return False, f"Installation error: {str(e)}"

    def segment_organs(self, sitk_image: sitk.Image,
                       fast: bool = True,
                       roi_subset: Optional[List[str]] = None) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Segment organs in the given SITK image using TotalSegmentator.

        Args:
            sitk_image: SimpleITK image to segment
            fast: Use fast mode (lower resolution but faster)
            roi_subset: List of specific organs to segment (None = all organs)

        Returns:
            (success, message, segmentation_array)
        """
        if not self.check_totalsegmentator_installed():
            return False, "TotalSegmentator not installed. Please install it first.", None

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, 'input.nii.gz')
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)

            try:
                # Save input image
                sitk.WriteImage(sitk_image, input_path)

                # Build command
                cmd = ['TotalSegmentator', '-i', input_path, '-o', output_dir]

                if fast:
                    cmd.append('--fast')

                if roi_subset:
                    cmd.extend(['--roi_subset'] + roi_subset)

                # Run segmentation
                print(f"Running TotalSegmentator: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode != 0:
                    return False, f"Segmentation failed: {result.stderr}", None

                # Load segmentation results
                segmentation_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]

                if not segmentation_files:
                    return False, "No segmentation output found", None

                # Combine all segmentation masks
                combined_mask = None
                self.detected_organs = []
                self.organ_bounds = {}

                for seg_file in segmentation_files:
                    organ_name = seg_file.replace('.nii.gz', '')
                    seg_path = os.path.join(output_dir, seg_file)
                    seg_image = sitk.ReadImage(seg_path)
                    seg_array = sitk.GetArrayFromImage(seg_image)

                    if np.any(seg_array > 0):
                        self.detected_organs.append(organ_name)

                        # Calculate bounding box
                        indices = np.where(seg_array > 0)
                        if len(indices[0]) > 0:
                            self.organ_bounds[organ_name] = {
                                'z_min': int(np.min(indices[0])),
                                'z_max': int(np.max(indices[0])),
                                'y_min': int(np.min(indices[1])),
                                'y_max': int(np.max(indices[1])),
                                'x_min': int(np.min(indices[2])),
                                'x_max': int(np.max(indices[2])),
                                'volume_voxels': int(np.sum(seg_array > 0))
                            }

                        # Combine masks (assign unique label to each organ)
                        if combined_mask is None:
                            combined_mask = np.zeros_like(seg_array, dtype=np.uint8)

                        organ_id = len(self.detected_organs)
                        combined_mask[seg_array > 0] = organ_id

                self.segmentation_mask = combined_mask

                message = f"Successfully detected {len(self.detected_organs)} organs: {', '.join(self.detected_organs)}"
                return True, message, combined_mask

            except subprocess.TimeoutExpired:
                return False, "Segmentation timeout (>10 minutes)", None
            except Exception as e:
                return False, f"Segmentation error: {str(e)}", None

    def get_organ_centroid(self, organ_name: str) -> Optional[Tuple[int, int, int]]:
        """Get the centroid (center) of a detected organ."""
        if organ_name not in self.organ_bounds:
            return None

        bounds = self.organ_bounds[organ_name]
        z_center = (bounds['z_min'] + bounds['z_max']) // 2
        y_center = (bounds['y_min'] + bounds['y_max']) // 2
        x_center = (bounds['x_min'] + bounds['x_max']) // 2

        return (z_center, y_center, x_center)

    def get_organ_overlay(self, slice_index: int, view: str,
                          alpha: float = 0.3) -> Optional[np.ndarray]:
        """
        Get a colored overlay of organs for a specific slice.

        Args:
            slice_index: Slice number
            view: 'axial', 'coronal', or 'sagittal'
            alpha: Transparency (0-1)

        Returns:
            RGBA overlay array or None
        """
        if self.segmentation_mask is None:
            return None

        # Extract slice based on view
        if view == 'axial':
            mask_slice = self.segmentation_mask[slice_index, :, :]
        elif view == 'coronal':
            mask_slice = self.segmentation_mask[:, slice_index, :]
        elif view == 'sagittal':
            mask_slice = self.segmentation_mask[:, :, slice_index]
        else:
            return None

        # Create RGBA overlay
        overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)

        # Color each organ
        for idx, organ_name in enumerate(self.detected_organs, start=1):
            mask = mask_slice == idx
            if np.any(mask):
                # Use predefined color or generate one
                if organ_name in self.ORGAN_INFO:
                    color = self.ORGAN_INFO[organ_name]['color']
                else:
                    # Generate color based on organ index
                    color = plt.cm.tab20(idx % 20)[:3]
                    color = tuple(int(c * 255) for c in color)

                overlay[mask, 0] = color[0] / 255.0
                overlay[mask, 1] = color[1] / 255.0
                overlay[mask, 2] = color[2] / 255.0
                overlay[mask, 3] = alpha

        return overlay

    def get_organ_statistics(self) -> Dict:
        """Get statistics about detected organs."""
        if not self.detected_organs:
            return {}

        stats = {}
        for organ_name in self.detected_organs:
            if organ_name in self.organ_bounds:
                bounds = self.organ_bounds[organ_name]
                stats[organ_name] = {
                    'volume_voxels': bounds['volume_voxels'],
                    'bounds': f"[{bounds['z_min']}-{bounds['z_max']}, "
                              f"{bounds['y_min']}-{bounds['y_max']}, "
                              f"{bounds['x_min']}-{bounds['x_max']}]",
                    'display_name': self.ORGAN_INFO.get(organ_name, {}).get('name', organ_name)
                }

        return stats

    def save_segmentation(self, output_path: str, original_image: sitk.Image) -> bool:
        """Save segmentation mask as NIfTI file with proper metadata."""
        if self.segmentation_mask is None:
            return False

        try:
            seg_image = sitk.GetImageFromArray(self.segmentation_mask)
            seg_image.CopyInformation(original_image)
            sitk.WriteImage(seg_image, output_path)
            return True
        except Exception as e:
            print(f"Error saving segmentation: {e}")
            return False


# Alternative: Using MONAI for organ detection (if TotalSegmentator is not available)
class MonaiOrganDetector:
    """
    Alternative organ detector using MONAI pre-trained models.
    Requires: pip install monai
    """

    def __init__(self):
        self.device = None
        self.model = None

    def check_monai_available(self) -> bool:
        """Check if MONAI is installed."""
        try:
            import monai
            return True
        except ImportError:
            return False

    def load_model(self, model_name: str = "spleen_ct_segmentation"):
        """
        Load a pre-trained MONAI model.
        Available models:
        - spleen_ct_segmentation
        - pancreas_ct_segmentation
        - liver_ct_segmentation
        """
        try:
            import torch
            from monai.networks.nets import UNet
            from monai.networks.layers import Norm

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model architecture (example for spleen)
            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            ).to(self.device)

            # Load pre-trained weights (you would need to download these)
            # self.model.load_state_dict(torch.load(model_path))

            return True
        except Exception as e:
            print(f"Error loading MONAI model: {e}")
            return False


import matplotlib.pyplot as plt  # Add this import at the top of the file