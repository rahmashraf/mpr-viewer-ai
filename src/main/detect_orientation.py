import os
import tensorflow as tf
import numpy as np
import pydicom  # Used for reading DICOM files

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to script location (src/main/)
MODEL_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'model', 'model.keras'))
CLASS_NAMES_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'model', 'class_names.txt'))
IMAGE_SIZE = (224, 224)

print(f"Loading model from: {MODEL_PATH}")
print(f"Loading class names from: {CLASS_NAMES_PATH}")

# --- Load the saved model and class names ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f]
    print(f"Model loaded successfully with {len(class_names)} classes")
except IOError as e:
    print(f"Error loading model or class names file: {e}")
    raise


# --- Function to predict a single image ---
def predict_image(image_path):
    """Loads an image, preprocesses it, and returns the predicted class and confidence."""
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence


# --- Function to predict a single DICOM image ---
def predict_dicom_image(pixel_array):
    """Loads a DICOM image, preprocesses it, and returns the predicted class and confidence."""

    # 2. Normalize pixel values to the 0-255 range
    if pixel_array.dtype != np.uint8:
        pixel_array = pixel_array.astype(float)
        # Ensure array has non-zero max to avoid division by zero
        if pixel_array.max() > 0:
            pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        pixel_array = pixel_array.astype(np.uint8)

    # 3. Convert grayscale to 3-channel RGB by duplicating the channel
    if len(pixel_array.shape) == 2:  # Check if it's a 2D (grayscale) image
        img_rgb = np.stack([pixel_array] * 3, axis=-1)
    else:
        img_rgb = pixel_array  # Assume it's already in a compatible format

    # 4. Resize the image to what the model expects
    img_resized = tf.image.resize(img_rgb, IMAGE_SIZE)

    # 5. Create a batch and make a prediction
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of 1

    predictions = model.predict(img_array, verbose=0)  # verbose=0 hides the progress bar
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence


if __name__ == '__main__':
    try:
        dicom_path = r"F:\CUFE-MPR\frontal\image-00001.dcm"
        print(f"\nReading DICOM file: {dicom_path}")

        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array

        print(f"Image shape: {pixel_array.shape}")
        print(f"Pixel value range: [{pixel_array.min()}, {pixel_array.max()}]")

        predicted_class, confidence = predict_dicom_image(pixel_array)
        print(f"\n✓ Prediction: This image is most likely a(n) {predicted_class} with {confidence:.2f}% confidence.")
    except FileNotFoundError as e:
        print(f"Error: DICOM file not found: {e}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback

        traceback.print_exc()
