import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="models/keras_ocr_float16_ctc.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Details : ", input_details)

# Function to preprocess image (resize to expected input size, e.g., 128x32 for CTC models)
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((200, 31), Image.LANCZOS)  # Adjust size based on model input
    img = np.expand_dims(np.array(img) / 255.0, axis=(0, -1)).astype(np.float32)
    return img

# Function to run inference
def recognize_text(image_path):
    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Note: Output decoding depends on the model's training (e.g., CTC decoding)
    # For simplicity, this assumes a raw output; you'll need a decoder (e.g., CTC greedy decoder) later
    return output_data

# Test on the first detected image (adjust path based on your runs/detect/predict/images)
image_path = "runs/detect/predict/plate1.jpg"  # Replace with an actual detected image
result = recognize_text(image_path)
print(f"OCR Result: {result}")