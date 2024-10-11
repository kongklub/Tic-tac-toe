import time  # Import the time module
import numpy as np
import pyautogui
import sounddevice as sd
import tensorflow as tf

# Load the Teachable Machine model
model_path = r"C:\Users\kongk\Downloads\model\soundclassifier_with_metadata.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Expected input shape: {input_details[0]['shape']}")

# Function to record audio from the microphone
def record_audio(duration=1, fs=44032):
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete")
    return audio_data.flatten()  # Flatten the array to 1D

# Function to classify audio using the TensorFlow Lite model
def classify_audio(audio_data):
    audio_data = np.array(audio_data, dtype=np.float32)

    # Ensure the audio length matches the model's expected input length
    expected_length = input_details[0]['shape'][1]  # 44032 samples
    if len(audio_data) > expected_length:
        audio_data = audio_data[:expected_length]  # Truncate
    elif len(audio_data) < expected_length:
        audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)), 'constant')  # Pad with zeros

    # Reshape the data to match the input shape of the model
    audio_data = np.expand_dims(audio_data, axis=0)  # Add batch dimension (1, length)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()

    # Get the output and return the predicted class and probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), output_data  # Return both predicted class and probabilities

# Function to perform an action based on the predicted class
def perform_action(predicted_class):
    # Define the coordinates for clicking (example coordinates
    if predicted_class == 0:
        print("กลางกลาง")
        pyautogui.moveTo(412, 433, duration=1)
        pyautogui.click()
    elif predicted_class == 1:
        print("กลางขวา")
        pyautogui.moveTo(577, 433, duration=1)  # Move right
        pyautogui.click()
    elif predicted_class == 2:
        print("กลางซ้าย")
        pyautogui.moveTo(234, 433, duration=1)  # Move left
        pyautogui.click()
    elif predicted_class == 3:
        print("บนกลาง")
        pyautogui.moveTo(412, 259, duration=1)  # Move up
        pyautogui.click()
    elif predicted_class == 4:
        print("บนขวา")
        pyautogui.moveTo(578, 259, duration=1)  # Move up right
        pyautogui.click()
    elif predicted_class == 5:
        print("บนซ้าย")
        pyautogui.moveTo(234, 259, duration=1)  # Move up left
        pyautogui.click()
    elif predicted_class == 6:
        print("ปิด")
        pyautogui.moveTo(646, 900, duration=1)  # Move down
        pyautogui.click()
    elif predicted_class == 7:
        print("ล่างกลาง")
        pyautogui.moveTo(412 + 590, duration=1)  # Move down
        pyautogui.click()
    elif predicted_class == 8:
        print("ล่างขวา")
        pyautogui.moveTo(578 , 590, duration=1)  # Move down right
        pyautogui.click()
    elif predicted_class == 9:
        print("ล่างซ้าย")
        pyautogui.moveTo(234, 590, duration=1)  # Move down left
        pyautogui.click()
    else:
        print("Unknown class")

# Main loop to start the process
while True:
    audio = record_audio(duration=1)  # Record for 2 seconds
    predicted_class, output_data = classify_audio(audio)

    # Set confidence threshol
    confidence_threshold = 0.7  # Adjust as needed
    if output_data[0][predicted_class] < confidence_threshold:
        perform_action(predicted_class)

    time.sleep(2)  # Add a small delay before the next recording
