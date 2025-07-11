import os
import json
import re
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


def force_paragraphs(text: str) -> str:
    """
    Add paragraph breaks after periods, exclamation points, and question marks.
    """
    if not text:
        return text
    text = re.sub(r"([.?!])(\s+)", r"\1\n\n", text)
    return text.strip()


def detect_language(message: str) -> str:
    """
    Detects the language of a user message using GPT.
    Returns the language name (e.g. 'Turkish').
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert language detector. "
                        "Detect the language of the user's message. "
                        "Respond ONLY with the name of the language, like 'Turkish', 'English', etc. "
                        "Do NOT explain anything else."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0,
            max_tokens=10
        )
        detected = response.choices[0].message.content.strip()
        return detected
    except Exception as e:
        print("Language detection error:", e)
        return "English"


def get_chat_response(
    message: str,
    user_id: str,
    image_url: Optional[str] = None
) -> str:
    """
    Generate a response using OpenAI ChatCompletion with memory context.
    Supports optional image_url for vision-enabled models.
    """

    # Detect language unless no message is provided
    user_language = detect_language(message) if message else "English"
    print(f"Detected language: {user_language}")

    # Load recent history
    history = load_memory(user_id)[-20:]

    # Build new, more assertive system prompt
    wiser_prompt = (
        f"You are DayDream AI, a highly skilled transformation coach and fitness expert. "
        f"Always reply in {user_language} unless the user explicitly asks for another language. "
        "When the user shares a statement or asks a question:\n\n"
        "1. Critically analyze whether the user's statement is scientifically correct, partially correct, or incorrect.\n"
        "2. Clearly explain **why** you agree or disagree, citing evidence, examples, or reasoning.\n"
        "3. Conclude with clear, practical recommendations the user can follow.\n"
        "4. Be honest and direct. If something is dangerous or unhealthy, warn the user politely but firmly.\n"
        "5. Break complex ideas into numbered steps. Always keep a warm, encouraging, and professional tone.\n"
        "6. If unsure, admit uncertainty rather than guessing.\n"
        "7. If the message contains an image, analyze it appropriately and tie the analysis into your response.\n"
    )

    messages_payload = [
        {
            "role": "system",
            "content": wiser_prompt,
        }
    ]

    # Inject memory entries
    for entry in history:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            messages_payload.append(entry)
        elif isinstance(entry, str):
            if entry.startswith("🧑:"):
                messages_payload.append({"role": "user", "content": entry[3:].strip()})
            elif entry.startswith("🤖:"):
                messages_payload.append({"role": "assistant", "content": entry[3:].strip()})

    # Add image message if provided
    if image_url:
        messages_payload.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is an image for you to analyze:"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        )

    # Append user message
    if message:
        messages_payload.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_payload,
            temperature=0.5,    # slightly higher for more confident and diverse answers
            max_tokens=500,
            stream=False
        )
        raw_reply = response.choices[0].message.content

        # Format the text to preserve paragraph breaks
        reply = force_paragraphs(raw_reply)

        # Save memory
        new_memory = history + [
            {"role": "user", "content": message or f"[sent image: {image_url}]"},
            {"role": "assistant", "content": reply}
        ]
        save_memory(user_id, new_memory)

        return reply

    except OpenAIError as oe:
        return f"[OpenAI Error] {oe}"
    except Exception as e:
        return f"[Error] {e}"
