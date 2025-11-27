
import torch
from TTS.api import TTS
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import queue
import threading
import time
import sys

# --- Debugging Function ---
def list_microphones():
    """Prints a list of available input microphones and their indices."""
    print("Available microphones:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # Filter for input devices
        if device['max_input_channels'] > 0:
            print(f"  [{i}] {device['name']}")
    return devices
# --- End of New Function ---

# ----------------------------
# Text-to-Speech (TTS) Engine
# ----------------------------

# Define available TTS models for different languages
TTS_MODELS = {
    "en": "tts_models/en/vctk/vits",  # English
    "hi": "tts_models/hi/vctk/vits"   # Hindi
}
tts_instances = {}  # Cache for initialized TTS models

def speak(text: str, lang: str = "en", speed: float = 1.2):
    """
    Convert text to speech offline using Coqui TTS for a specified language.
    This is a blocking call.
    """
    global tts_instances
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Check if the TTS model for the specified language is already initialized
        if lang not in tts_instances:
            model_name = TTS_MODELS.get(lang)
            if not model_name:
                print(f"No TTS model found for language: {lang}")
                return
            print(f"Initializing TTS model for {lang}...")
            tts_instances[lang] = TTS(model_name).to(device)
            print("TTS model initialized.")

        tts = tts_instances[lang]
        temp_file = "temp_output.wav"
        tts.tts_to_file(text=text, file_path=temp_file, speed=speed)
        data, fs = sf.read(temp_file, dtype='float32')
        sd.play(data, fs)
        sd.wait()
        os.remove(temp_file)
    except Exception as e:
        print(f"Error in TTS engine for language {lang}: {e}")

# ----------------------------
# Speech-to-Text (STT) Engine
# ----------------------------

try:
    whisper_model = whisper.load_model("small")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None


def take_command(timeout: int = 5, device_index: int = None, language: str = "en") -> str:
    """
    Listens to user voice input and converts it to text (offline).
    Returns recognized text, or "none" if not understood.
    Accepts an optional device_index and language for transcription.
    """
    if whisper_model is None:
        print("Whisper recognizer is not available.")
        return "none"

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
            device=device_index
        ):
            print(f"🎤 Listening in {language}...")
            
            start_time = time.time()
            audio_data = []
            while time.time() - start_time < timeout:
                try:
                    data = q.get(block=False)
                    audio_data.append(np.frombuffer(data, dtype=np.int16))
                except queue.Empty:
                    time.sleep(0.1)
            
            if not audio_data:
                return "none"

            # Concatenate all audio chunks
            audio_np = np.concatenate(audio_data, axis=0).astype(np.float32) / 32768.0

            # Transcribe
            result = whisper_model.transcribe(audio_np, language=language, fp16=torch.cuda.is_available())
            return result.get("text", "").lower()

    except Exception as e:
        print(f"❌ Error during microphone input: {e}")
        print("Please check your microphone and its settings.")
        return "none"

if __name__ == "__main__":
    print("Listing microphones for debugging:")
    list_microphones()
    print("-" * 20)

    MIC_TO_USE = None

    # --- Test Hindi ---
    print("Say something in Hindi...")
    query_hi = take_command(timeout=8, device_index=MIC_TO_USE, language="hi")
    print("You said (Hindi):", query_hi)
    speak(f"आपने कहा: {query_hi}", lang="hi")

    print("-" * 20)

    # --- Test English ---
    print("Say something in English...")
    query_en = take_command(timeout=8, device_index=MIC_TO_USE, language="en")
    print("You said (English):", query_en)
    speak(f"You said: {query_en}", lang="en")
