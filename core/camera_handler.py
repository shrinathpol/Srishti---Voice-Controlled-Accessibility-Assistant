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
    print("Attempting to start live assistance...")
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
        speak("I can't start the live assistance because the detection model is missing.")
        return

    cap = None
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully.")
        
        # Attempt to find a working camera
        print("Searching for a webcam...")
        for i in range(5):
            print(f"Probing camera index {i}...")
            cap = cv2.VideoCapture(i)
            if cap and cap.isOpened():
                print(f"Success! Webcam found and opened at index {i}.")
                break
            else:
                print(f"Camera index {i} failed to open or is not available.")
                if cap:
                    cap.release() # Release cap if it was created but not opened
        
        if not cap or not cap.isOpened():
            speak("I could not open the webcam. Please ensure it is connected and not in use by another application.")
            raise IOError("Could not open webcam. Please ensure it's connected and not in use.")

        last_detection_time = 0
        cv2.namedWindow("Live Assistance", cv2.WINDOW_NORMAL)
        print("Starting main camera loop...")
        frame_count = 0

        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to grab frame. Retrying...")
                time.sleep(0.5) # A slightly longer pause might help with driver issues
                continue
            
            frame_count += 1
            if frame_count % 30 == 0: # Log every 30 frames
                print(f"Successfully processed {frame_count} frames.")

            # Perform inference
            results = model(frame, verbose=False)
            
            # Annotate the frame with detection results
            annotated_frame = results[0].plot()

            # Process detections
            detected_classes = {model.names[int(box.cls[0])] for result in results for box in result.boxes if box.cls}

            if detected_classes:
                # For simplicity, announce the first detected class, then apply cooldown
                last_detection_time = handle_button_detection(list(detected_classes)[0], last_detection_time)
            
            # Display the output
            cv2.imshow("Live Assistance", annotated_frame)
            
            # Check for 'q' to quit, non-blocking
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, stopping live assistance.")
                break
    
    except (IOError, Exception) as e:
        print(f"An error occurred during live assistance: {e}")
        speak("An unexpected error occurred with the camera. Please check the console for details.")

    finally:
        print("Closing down live assistance...")
        if cap and cap.isOpened():
            cap.release()
            print("Webcam released.")
        cv2.destroyAllWindows()
        print("All OpenCV windows destroyed.")
        # Ensure the stop event is set to terminate any related threads
        if not stop_event.is_set():
            stop_event.set()
        print("Live assistance has stopped.")
