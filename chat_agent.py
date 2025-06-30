import os
import json
from typing import Optional
from openai import OpenAI, OpenAIError

# Memory storage directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "chat_memories")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def memory_file_path(user_id: str) -> str:
    return os.path.join(MEMORY_DIR, f"{user_id}.json")


def load_memory(user_id: str) -> list:
    # Ensure memory directory exists
    os.makedirs(MEMORY_DIR, exist_ok=True)
    path = memory_file_path(user_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = f.read().strip()
            if not data:
                return []
            return json.loads(data)
    except (json.JSONDecodeError, IOError):
        # Reset corrupt memory
        with open(path, "w") as f:
            json.dump([], f)
        return []


def save_memory(user_id: str, messages: list) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(memory_file_path(user_id), "w") as f:
        json.dump(messages, f)


def get_chat_response(
    message: str,
    user_id: str,
    image_url: Optional[str] = None
) -> str:
    """
    Generate a response using OpenAI ChatCompletion with memory context.
    Supports optional image_url for vision-enabled models.
    """
    # Load recent history
    history = load_memory(user_id)[-20:]

    # Build system prompt
    messages_payload = [
        {
            "role": "system",
            "content": (
                "You are DayDream AI, a friendly, expert transformation coach. "
                "You can see and reason about images when provided. "
                "Respond with clear, step-by-step guidance and ask questions as needed."
            ),
        }
    ]

    # Inject memory entries (dicts or legacy strings)
    for entry in history:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            messages_payload.append(entry)
        elif isinstance(entry, str):
            if entry.startswith("🧑:"):
                messages_payload.append({"role": "user", "content": entry[3:].strip()})
            elif entry.startswith("🤖:"):
                messages_payload.append({"role": "assistant", "content": entry[3:].strip()})

    # If an image was provided, embed it in the payload
    if image_url:
        messages_payload.append(
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": "Here is an image for you to analyze:"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        )

    # Append the actual text message if present
    if message:
        messages_payload.append({"role": "user", "content": message})

    try:
        # Use a vision-enabled GPT model (e.g., gpt-4-vision-preview or gpt-4o)
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages_payload,
            temperature=0.7,
            max_tokens=500,
            stream=False
        )
        reply = response.choices[0].message.content

        # Update memory and save as dict entries
        new_memory = history + [
            {"role": "user",      "content": message or f"[sent image: {image_url}]"},
            {"role": "assistant", "content": reply}
        ]
        save_memory(user_id, new_memory)

        return reply

    except OpenAIError as oe:
        return f"[OpenAI Error] {oe}"
    except Exception as e:
        return f"[Error] {e}"
