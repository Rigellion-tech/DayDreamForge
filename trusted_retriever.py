import os
import json

# Path to your trusted data file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRUSTED_DATA_PATH = os.path.join(BASE_DIR, "trusted_data.json")

def get_trusted_context(query: str) -> str:
    """
    Searches a local JSON file for trusted medical/fitness context
    matching the user's query.

    Returns a string with trusted info if found, otherwise empty string.
    """

    if not query:
        return ""

    try:
        if not os.path.exists(TRUSTED_DATA_PATH):
            # File does not exist
            return ""

        with open(TRUSTED_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Very simple matching:
        for item in data:
            topic = item.get("topic", "").lower()
            if topic and topic in query.lower():
                text = item.get("text", "")
                source = item.get("source", "")
                return f"{text} (Source: {source})"

        return ""

    except Exception as e:
        print("Trusted context lookup error:", e)
        return ""
