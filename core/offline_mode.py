# Jarvis/core/offline_mode.py

import datetime
import json
from .speech_engine import speak

CACHE_FILE = 'data/online_cache.json'

def handle_offline_command(query):
    """Handles a command by checking cache, then using local logic."""
    
    # 1. Check if the command is in the cache
    cached_response = check_cache(query)
    if cached_response:
        speak(f"I remember you asked that before. Here is the answer: {cached_response}")
        return
        
    # 2. If not in cache, fall back to keyword-based logic
    if 'what is the time' in query: # query is now the command itself, e.g., "what is the time"
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        speak(f"The time is {strTime}")
    
    elif 'hello jarvis' in query:
        # This combines the logic from your two duplicate blocks
        speak("Hello! I am currently in offline mode. How can I help?")
    
    else:
        speak("I am currently in offline mode and can only perform a few simple tasks. Please try a different command.")

def check_cache(query):
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            return cache.get(query.lower())
    except (FileNotFoundError, json.JSONDecodeError):
        return None