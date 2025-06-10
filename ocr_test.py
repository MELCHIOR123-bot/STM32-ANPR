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

char_map = {i: c for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')}
char_map[-1] = ''  # Padding/end token
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
    # Simple CTC greedy decoding: take the max index per timestep and filter out -1
    decoded = [char_map[np.argmax(output_data[0][t])] for t in range(output_data.shape[1]) if np.argmax(output_data[0][t]) != -1]
    return ''.join(decoded)
# Test on the first detected image (adjust path based on your runs/detect/predict/images)
image_path = "runs/detect/predict/plate1.jpg"  # Replace with an actual detected image
result = recognize_text(image_path)
print(f"OCR Result: {result}")