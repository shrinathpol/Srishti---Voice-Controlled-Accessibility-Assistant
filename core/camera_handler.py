import cv2
from ultralytics import YOLO
import os
import time
import threading
from core.speech_engine import speak
from .config import YOLO_MODEL_PATH, DETECTION_COOLDOWN

def handle_button_detection(button_class, last_detection_time):
    """Handles the event of a button detection with a cooldown."""
    current_time = time.time()
    if (current_time - last_detection_time) > DETECTION_COOLDOWN:
        speak(f"I see a {button_class}. What would you like to do?")
        return current_time
    return last_detection_time

def run_live_assistance(stop_event: threading.Event):
    """Runs the live assistance mode with camera and YOLO detection."""
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
        speak("I can't start the live assistance because the detection model is missing.")
        return

    try:
        model = YOLO(YOLO_MODEL_PATH)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            speak("I'm having trouble accessing the webcam.")
            return
        
        last_detection_time = 0

        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame from webcam.")
                time.sleep(1) # Wait a bit before trying again
                continue

            results = model(frame, verbose=False) # Set verbose to False to reduce console spam

            detected_classes = set()
            for result in results:
                for box in result.boxes:
                    if box.cls:
                        class_name = model.names[int(box.cls[0])]
                        detected_classes.add(class_name)

            if detected_classes:
                # Announce the first button found, then cooldown
                last_detection_time = handle_button_detection(list(detected_classes)[0], last_detection_time)

            # Reduce the frame processing rate
            time.sleep(0.5)

    except Exception as e:
        print(f"An error occurred during live assistance: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print("Live assistance resources released.")
