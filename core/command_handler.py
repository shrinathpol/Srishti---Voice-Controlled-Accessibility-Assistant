import json
import google.generativeai as genai
import os
from .config import GEMINI_API_KEY

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the generative model
try:
    model = genai.GenerativeModel('gemini-2.5-pro')
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    model = None

# Create a global chat session object
if model:
    chat_session = model.start_chat(history=[])
else:
    chat_session = None

CACHE_FILE = 'data/online_cache.json'

def save_to_cache(query, response):
    """Saves a query-response pair to the cache file."""
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    
    cache[query.lower()] = response
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def get_gemini_response(user_input):
    """Sends a query and conversation history to Gemini and returns the AI's response."""
    
    if not chat_session:
        return "Gemini model is not available. Please check the logs for errors."

    # --- FIX 2: Check for empty input BEFORE calling the API ---
    if not user_input or user_input.strip() == "":
        print("INFO: Empty input received. Skipping API call.")
        return "I didn't catch that. Could you please repeat?"
        
    try:
        # Use the chat_session.send_message method
        # This automatically appends the user input and the model's response to the history
        response = chat_session.send_message(user_input)
        
        # Check if the response contains text
        if response and hasattr(response, 'text'):
            save_to_cache(user_input, response.text) # Cache the response
            return response.text 
        else:
            return "I couldn't generate a text response for that."
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "I am sorry, I am unable to process that request at the moment."

        