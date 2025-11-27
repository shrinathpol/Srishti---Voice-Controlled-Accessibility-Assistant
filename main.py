import datetime
import requests
import os
import sys
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_VERBOSITY"] = "ERROR"

# Import core modules
from core.speech_engine import speak, take_command
from core.command_handler import get_gemini_response, CACHE_FILE as ONLINE_CACHE_PATH
from core.offline_mode import handle_offline_command
from offline_model_trainer.src.offline_inference import load_model, preprocess_input, make_prediction
from core.camera_handler import run_live_assistance
from core.config import OFFLINE_MODEL_PATH, DETECTION_COOLDOWN, YOLO_MODEL_PATH

# --- Constants ---
VALIDATION_DATA_PATH = os.path.join('offline_model_trainer', 'data', 'validation_data.json')
NEW_TRAINING_DATA_PATH = os.path.join('data', 'knowledge_base', 'new_training_data.json')
SIMILARITY_THRESHOLD = 0.6


class Srishti:
    def __init__(self):
        self.language = "en-us"
        self.offline_model = None
        self.sbert_model = None
        self.offline_model_loading = False
        self.sbert_model_loading = False
        self.live_assistance_thread = None
        self.stop_live_assistance_event = threading.Event()
        self.speech_speed = 1.3
        self.validation_data = []

    def load_sbert_model_background(self):
        """Loads the SentenceTransformer model in a background thread."""
        self.sbert_model_loading = True
        print("Loading SentenceTransformer model in the background...")
        try:
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
        finally:
            self.sbert_model_loading = False

    def load_offline_classifier_background(self):
        """Loads the offline classification model in a background thread."""
        self.offline_model_loading = True
        print("Loading offline classification model in the background...")
        try:
            if os.path.exists(OFFLINE_MODEL_PATH):
                self.offline_model = load_model(OFFLINE_MODEL_PATH)
                print("Offline classification model loaded successfully.")
            else:
                print(f"Warning: Offline model file not found at {OFFLINE_MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load offline classification model. Reason: {e}")
        finally:
            self.offline_model_loading = False

    def is_connected(self):
        """Check for an active internet connection."""
        try:
            requests.get('https://www.google.com', timeout=1.5)
            return True
        except requests.ConnectionError:
            return False

    def ensure_cache_file_exists(self):
        """Checks if the cache directory and file exist, creating them if not."""
        cache_dir = os.path.dirname(ONLINE_CACHE_PATH)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.exists(ONLINE_CACHE_PATH):
            with open(ONLINE_CACHE_PATH, 'w') as f:
                json.dump({}, f)

    def load_validation_data(self):
        """Load validation data from a JSON file."""
        if not os.path.exists(VALIDATION_DATA_PATH):
            print(f"Warning: Validation data file not found at {VALIDATION_DATA_PATH}")
            self.validation_data = []
        else:
            with open(VALIDATION_DATA_PATH, 'r') as f:
                self.validation_data = json.load(f)

    def save_for_training(self, query, response):
        """Saves the query and response for later training."""
        if not os.path.exists(NEW_TRAINING_DATA_PATH):
            data = []
        else:
            with open(NEW_TRAINING_DATA_PATH, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        
        data.append({"input": query, "expected_output": response})
        
        with open(NEW_TRAINING_DATA_PATH, 'w') as f:
            json.dump(data, f, indent=4)

    def manage_speech(self, text):
        """A wrapper for the speak function."""
        # speak(text, speed=self.speech_speed)
        pass

    def find_best_match_by_similarity(self, query, threshold=0.6):
        """Finds the best matching response from validation data using cosine similarity."""
        if not self.sbert_model or not self.validation_data:
            return None

        best_match = None
        max_similarity = 0
        query_embedding = self.sbert_model.encode(query)

        for item in self.validation_data:
            item_embedding = self.sbert_model.encode(item['input'])
            similarity = cosine_similarity([query_embedding], [item_embedding])[0][0]
            if similarity > max_similarity and similarity > threshold:
                max_similarity = similarity
                best_match = item['expected_output']
        return best_match

    def get_offline_response(self, query):
        """
        Generates an offline response using a tiered approach.
        Returns a special string if models are still loading.
        """
        # Tier 1: Use the trained ML model
        if self.offline_model:
            preprocessed_query = preprocess_input({'input': query})
            prediction = make_prediction(self.offline_model, [preprocessed_query])[0]
            if prediction != "unknown":
                return prediction
        elif self.offline_model_loading:
            return "loading"

        # Tier 2: Use sentence similarity as a fallback
        if self.sbert_model:
            best_match = self.find_best_match_by_similarity(query)
            if best_match:
                return best_match
        elif self.sbert_model_loading:
            return "loading"

        # Tier 3: Default response
        return None

    def wish_me(self):
        """Greets the user based on the time of day."""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            greeting = "Good Morning!"
        elif 12 <= hour < 18:
            greeting = "Good Afternoon!"
        else:
            greeting = "Good Evening!"
        self.manage_speech(f"{greeting} I am Srishti. How may I assist you?")

    def handle_online_query(self, query):
        """Handles online queries using the Gemini API."""
        try:
            print("Processing online...")
            response_text = get_gemini_response(query)
            self.save_for_training(query, response_text)
            return response_text
        except Exception as e:
            print(f"Error from online API: {e}")
            return "I'm having trouble connecting to my online services."

    def handle_offline_query(self, query):
        """Handles offline queries using the offline model and sentence similarity."""
        print("Processing offline...")
        offline_command = self.get_offline_response(query)
        if offline_command == "loading":
            return "My offline capabilities are still initializing. Please try again in a moment."
        elif offline_command:
            handle_offline_command(offline_command)
            return ""  # Return an empty string as the command is handled
        return "I'm not sure how to handle that offline."

    def shutdown(self):
        """Shuts down the application, stopping background threads."""
        print("Shutting down application...")
        self.stop_live_assistance_event.set()
        if self.live_assistance_thread and self.live_assistance_thread.is_alive():
            self.live_assistance_thread.join()
        print("Application shut down.")

    def start_live_assistance(self):
        """Starts the live assistance thread."""
        if self.live_assistance_thread and self.live_assistance_thread.is_alive():
            self.manage_speech("Live assistance is already running.")
            return

        self.stop_live_assistance_event.clear()
        thread = threading.Thread(target=run_live_assistance, args=(self.stop_live_assistance_event,))
        thread.daemon = True
        thread.start()
        self.live_assistance_thread = thread
        self.manage_speech("Live assistance has been started.")

    def stop_live_assistance(self):
        """Stops the live assistance thread."""
        if not self.live_assistance_thread or not self.live_assistance_thread.is_alive():
            self.manage_speech("Live assistance is not currently running.")
            return

        self.manage_speech("Stopping live assistance.")
        self.stop_live_assistance_event.set()
        # Wait for the thread to finish
        self.live_assistance_thread.join(timeout=5)
        self.live_assistance_thread = None
        self.manage_speech("Live assistance has been stopped.")

    def initialize_app(self):
        """Initializes the application, loading models and data."""
        self.ensure_cache_file_exists()
        self.wish_me()

        # Start loading models in the background
        threading.Thread(target=self.load_sbert_model_background, daemon=True).start()
        threading.Thread(target=self.load_offline_classifier_background, daemon=True).start()

        # Load validation data
        self.load_validation_data()

    def process_query(self, query):
        """Processes the user's query."""
        if not query or query.strip() == "none":
            return ""

        if self.is_connected():
            response_text = self.handle_online_query(query)
        else:
            response_text = self.handle_offline_query(query)
        return response_text

    def run(self):
        """Main function to run the Srishti assistant."""
        self.initialize_app()
        
        while True:
            query = take_command().lower()
            print(f"You said: {query}")

            if query in ["goodbye", "exit", "quit"]:
                self.manage_speech("Goodbye!")
                break
            
            # Handle live assistance commands
            if "start live assistance" in query:
                self.start_live_assistance()
                continue
            elif "stop live assistance" in query:
                self.stop_live_assistance()
                continue

            response_text = self.process_query(query)
            
            if response_text:
                self.manage_speech(response_text)
                
        self.shutdown()

if __name__ == "__main__":
    app = Srishti()
    app.run()