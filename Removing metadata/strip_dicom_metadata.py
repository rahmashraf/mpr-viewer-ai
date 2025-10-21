import os
import argparse
import pydicom
import shutil


def strip_orientation_metadata(input_path, output_path):
    """
    Remove ImageOrientationPatient metadata from DICOM file(s)
    to force the detection code to use volume-based heuristic.
    """
    if os.path.isfile(input_path):
        # Single file
        strip_single_file(input_path, output_path)
    elif os.path.isdir(input_path):
        # Directory of DICOM files
        os.makedirs(output_path, exist_ok=True)
        dcm_files = [f for f in os.listdir(input_path) if f.lower().endswith('.dcm')]

        print(f"Found {len(dcm_files)} DICOM files to process...")

        for filename in dcm_files:
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)
            strip_single_file(input_file, output_file)

        print(f"✓ Processed {len(dcm_files)} files")
        print(f"✓ Stripped DICOMs saved to: {output_path}")
    else:
        raise ValueError("Input must be a DICOM file or directory")


def strip_single_file(input_file, output_file):
    """Strip metadata from a single DICOM file"""
    try:
        ds = pydicom.dcmread(input_file)

        # Remove orientation metadata tags
        tags_to_remove = [
            'ImageOrientationPatient',
            'ImagePositionPatient',  # Optional: also remove position info
        ]

        removed = []
        for tag_name in tags_to_remove:
            if hasattr(ds, tag_name):
                delattr(ds, tag_name)
                removed.append(tag_name)

        # Save the modified DICOM
        ds.save_as(output_file)

        if removed:
            print(f"  {os.path.basename(input_file)}: Removed {', '.join(removed)}")
        else:
            print(f"  {os.path.basename(input_file)}: No orientation metadata found (already stripped)")

    except Exception as e:
        print(f"  ERROR processing {input_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Strip orientation metadata from DICOM files for testing volume-based detection"
    )
    parser.add_argument(
        "--input", "-i",
        default=r"C:\Users\Youssef\Desktop\mprvoew12\mpr-viewer-ai\ID_0000_AGE_0060_CONTRAST_1_CT.dcm",
        help="Input DICOM file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        default=r"C:\Users\Youssef\Desktop\mprvoew12\stripped.dcm",
        help="Output DICOM file or directory"
    )

    args = parser.parse_args()

    print("=== Stripping DICOM Orientation Metadata ===")
    strip_orientation_metadata(args.input, args.output)
    print("\n✓ Done! Now run detect_orientation.py on the output to test volume-based detection.")


if __name__ == "__main__":
    main()