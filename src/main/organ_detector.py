"""
Organ Detection Module for MRI/CT Viewer
Detects body parts and organs from DICOM metadata
"""

import pydicom
import logging
from typing import Tuple, Optional, Dict
import re


class OrganDetector:
    """Detects organs and body parts from DICOM metadata"""

    # Comprehensive mapping of DICOM body part codes to organ names
    BODY_PART_MAPPING = {
        # Head and Brain
        'BRAIN': ('Brain', '🧠'),
        'HEAD': ('Head/Brain', '🧠'),
        'SKULL': ('Skull', '💀'),
        'CSKULL': ('Skull', '💀'),
        'SINUS': ('Sinuses', '👃'),
        'ORBIT': ('Orbit/Eye', '👁️'),
        'EYE': ('Eye', '👁️'),
        'EAR': ('Ear', '👂'),
        'FACE': ('Face', '😊'),
        'JAW': ('Jaw', '🦷'),
        'TMJOINT': ('TMJ Joint', '🦷'),

        # Spine
        'CSPINE': ('Cervical Spine', '🦴'),
        'TSPINE': ('Thoracic Spine', '🦴'),
        'LSPINE': ('Lumbar Spine', '🦴'),
        'SSPINE': ('Sacral Spine', '🦴'),
        'SPINE': ('Spine', '🦴'),
        'WHOLESPINE': ('Whole Spine', '🦴'),

        # Thorax
        'CHEST': ('Chest', '🫁'),
        'THORAX': ('Thorax', '🫁'),
        'LUNG': ('Lung', '🫁'),
        'HEART': ('Heart', '❤️'),
        'CLAVICLE': ('Clavicle', '🦴'),
        'RIB': ('Ribs', '🦴'),
        'STERNUM': ('Sternum', '🦴'),
        'MEDIASTINUM': ('Mediastinum', '🫁'),

        # Abdomen
        'ABDOMEN': ('Abdomen', '🔶'),
        'LIVER': ('Liver', '🟤'),
        'KIDNEY': ('Kidney', '🫘'),
        'SPLEEN': ('Spleen', '🟣'),
        'PANCREAS': ('Pancreas', '🟡'),
        'GALLBLADDER': ('Gallbladder', '🟢'),
        'STOMACH': ('Stomach', '🔴'),
        'BOWEL': ('Bowel', '🟠'),
        'COLON': ('Colon', '🟠'),

        # Pelvis
        'PELVIS': ('Pelvis', '🦴'),
        'HIP': ('Hip', '🦴'),
        'PROSTATE': ('Prostate', '🔵'),
        'UTERUS': ('Uterus', '🟣'),
        'OVARY': ('Ovary', '🟣'),
        'BLADDER': ('Bladder', '🔵'),

        # Extremities
        'SHOULDER': ('Shoulder', '💪'),
        'HUMERUS': ('Humerus', '🦴'),
        'ELBOW': ('Elbow', '🦴'),
        'FOREARM': ('Forearm', '💪'),
        'WRIST': ('Wrist', '✋'),
        'HAND': ('Hand', '✋'),
        'FINGER': ('Finger', '👆'),
        'THUMB': ('Thumb', '👍'),

        'FEMUR': ('Femur', '🦴'),
        'KNEE': ('Knee', '🦵'),
        'TIBIA': ('Tibia', '🦴'),
        'FIBULA': ('Fibula', '🦴'),
        'ANKLE': ('Ankle', '🦶'),
        'FOOT': ('Foot', '🦶'),
        'TOE': ('Toe', '🦶'),

        # Neck and Throat
        'NECK': ('Neck', '🦒'),
        'THYROID': ('Thyroid', '🦋'),
        'LARYNX': ('Larynx', '🗣️'),
        'PHARYNX': ('Pharynx', '🗣️'),

        # Vascular
        'AORTA': ('Aorta', '❤️'),
        'CAROTID': ('Carotid Artery', '❤️'),
        'VESSEL': ('Blood Vessel', '❤️'),

        # Other
        'BREAST': ('Breast', '👙'),
        'ADRENAL': ('Adrenal Gland', '🟡'),
    }

    # Series description keywords for organ detection
    SERIES_KEYWORDS = {
        'brain': ('Brain', '🧠'),
        'head': ('Head/Brain', '🧠'),
        'cerebr': ('Brain', '🧠'),
        'cardiac': ('Heart', '❤️'),
        'heart': ('Heart', '❤️'),
        'liver': ('Liver', '🟤'),
        'hepat': ('Liver', '🟤'),
        'renal': ('Kidney', '🫘'),
        'kidney': ('Kidney', '🫘'),
        'lung': ('Lung', '🫁'),
        'pulmon': ('Lung', '🫁'),
        'spine': ('Spine', '🦴'),
        'vertebr': ('Spine', '🦴'),
        'pelv': ('Pelvis', '🦴'),
        'abdom': ('Abdomen', '🔶'),
        'chest': ('Chest', '🫁'),
        'thorax': ('Thorax', '🫁'),
        'knee': ('Knee', '🦵'),
        'shoulder': ('Shoulder', '💪'),
        'hip': ('Hip', '🦴'),
        'hand': ('Hand', '✋'),
        'wrist': ('Wrist', '✋'),
        'foot': ('Foot', '🦶'),
        'ankle': ('Ankle', '🦶'),
        'elbow': ('Elbow', '🦴'),
        'prostat': ('Prostate', '🔵'),
        'breast': ('Breast', '👙'),
        'mamm': ('Breast', '👙'),
    }

    @staticmethod
    def detect_organ(dicom_path: str) -> Tuple[str, str, float, Dict[str, str]]:
        """
        Detect organ from DICOM metadata

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Tuple of (organ_name, emoji, confidence, metadata_dict)
        """
        try:
            ds = pydicom.dcmread(dicom_path)
            metadata = OrganDetector._extract_metadata(ds)

            # Try multiple detection methods
            organ, emoji, confidence = OrganDetector._detect_from_body_part(ds)
            if confidence < 0.8:
                organ2, emoji2, conf2 = OrganDetector._detect_from_series_description(ds)
                if conf2 > confidence:
                    organ, emoji, confidence = organ2, emoji2, conf2

            if confidence < 0.6:
                organ3, emoji3, conf3 = OrganDetector._detect_from_study_description(ds)
                if conf3 > confidence:
                    organ, emoji, confidence = organ3, emoji3, conf3

            # Add anatomical region info
            if confidence < 0.5:
                organ = "Unknown Region"
                emoji = "❓"
                confidence = 0.3

            return organ, emoji, confidence, metadata

        except Exception as e:
            logging.error(f"Error detecting organ: {e}")
            return "Unknown", "❓", 0.0, {}

    @staticmethod
    def _extract_metadata(ds: pydicom.Dataset) -> Dict[str, str]:
        """Extract relevant metadata from DICOM"""
        metadata = {}

        # Basic patient info
        metadata['Patient Name'] = str(getattr(ds, 'PatientName', 'N/A'))
        metadata['Patient ID'] = str(getattr(ds, 'PatientID', 'N/A'))
        metadata['Patient Sex'] = str(getattr(ds, 'PatientSex', 'N/A'))
        metadata['Patient Age'] = str(getattr(ds, 'PatientAge', 'N/A'))

        # Study info
        metadata['Study Date'] = str(getattr(ds, 'StudyDate', 'N/A'))
        metadata['Study Description'] = str(getattr(ds, 'StudyDescription', 'N/A'))
        metadata['Series Description'] = str(getattr(ds, 'SeriesDescription', 'N/A'))
        metadata['Body Part Examined'] = str(getattr(ds, 'BodyPartExamined', 'N/A'))

        # Modality info
        metadata['Modality'] = str(getattr(ds, 'Modality', 'N/A'))
        metadata['Manufacturer'] = str(getattr(ds, 'Manufacturer', 'N/A'))
        metadata['Station Name'] = str(getattr(ds, 'StationName', 'N/A'))

        # Protocol
        metadata['Protocol Name'] = str(getattr(ds, 'ProtocolName', 'N/A'))
        metadata['Sequence Name'] = str(getattr(ds, 'SequenceName', 'N/A'))

        return metadata

    @staticmethod
    def _detect_from_body_part(ds: pydicom.Dataset) -> Tuple[str, str, float]:
        """Detect organ from BodyPartExamined tag"""
        if not hasattr(ds, 'BodyPartExamined'):
            return "Unknown", "❓", 0.0

        body_part = str(ds.BodyPartExamined).upper().strip()

        # Direct match
        if body_part in OrganDetector.BODY_PART_MAPPING:
            organ, emoji = OrganDetector.BODY_PART_MAPPING[body_part]
            return organ, emoji, 0.95

        # Partial match
        for key, (organ, emoji) in OrganDetector.BODY_PART_MAPPING.items():
            if key in body_part or body_part in key:
                return organ, emoji, 0.85

        return body_part.title(), "🔍", 0.5

    @staticmethod
    def _detect_from_series_description(ds: pydicom.Dataset) -> Tuple[str, str, float]:
        """Detect organ from SeriesDescription"""
        if not hasattr(ds, 'SeriesDescription'):
            return "Unknown", "❓", 0.0

        series_desc = str(ds.SeriesDescription).lower()

        for keyword, (organ, emoji) in OrganDetector.SERIES_KEYWORDS.items():
            if keyword in series_desc:
                return organ, emoji, 0.75

        return "Unknown", "❓", 0.0

    @staticmethod
    def _detect_from_study_description(ds: pydicom.Dataset) -> Tuple[str, str, float]:
        """Detect organ from StudyDescription"""
        if not hasattr(ds, 'StudyDescription'):
            return "Unknown", "❓", 0.0

        study_desc = str(ds.StudyDescription).lower()

        for keyword, (organ, emoji) in OrganDetector.SERIES_KEYWORDS.items():
            if keyword in study_desc:
                return organ, emoji, 0.65

        return "Unknown", "❓", 0.0

    @staticmethod
    def format_detection_report(organ: str, emoji: str, confidence: float,
                                metadata: Dict[str, str]) -> str:
        """Format a readable detection report"""
        report = []
        report.append("=" * 50)
        report.append("ORGAN DETECTION REPORT")
        report.append("=" * 50)
        report.append(f"\n{emoji} Detected Organ: {organ}")
        report.append(f"📊 Confidence: {confidence * 100:.1f}%")
        report.append("\n" + "-" * 50)
        report.append("METADATA:")
        report.append("-" * 50)

        for key, value in metadata.items():
            if value != 'N/A' and value.strip():
                report.append(f"{key:.<30} {value}")

        report.append("=" * 50)
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample DICOM file
    import sys

    if len(sys.argv) > 1:
        dicom_path = sys.argv[1]
        organ, emoji, confidence, metadata = OrganDetector.detect_organ(dicom_path)
        report = OrganDetector.format_detection_report(organ, emoji, confidence, metadata)
        print(report)
    else:
        print("Usage: python organ_detector.py <dicom_file_path>")