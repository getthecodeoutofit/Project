# settings/preferences.py

import json

PREFERENCES_FILE = "preferences.json"

def load_preferences():
    try:
        with open(PREFERENCES_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"font": "Courier", "font_size": 12, "font_color": "#000000"}

def save_preferences(preferences):
    with open(PREFERENCES_FILE, "w") as file:
        json.dump(preferences, file)
