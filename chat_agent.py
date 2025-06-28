import os
import json
from openai import OpenAI, OpenAIError

# Memory storage directory
MEMORY_DIR = "chat_memories"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def memory_file_path(user_id):
    return os.path.join(MEMORY_DIR, f"{user_id}.json")


def load_memory(user_id):
    try:
        with open(memory_file_path(user_id), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_memory(user_id, messages):
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(memory_file_path(user_id), "w") as f:
        json.dump(messages, f)


def get_chat_response(message: str, user_id: str) -> str:
    """
    Generate a response using OpenAI ChatCompletion with memory context.
    """
    # Load recent history (last 20 entries)
    history = load_memory(user_id)[-20:]

    # Build messages for OpenAI
    messages_payload = [
        {
            "role": "system",
            "content": (
                "You are DayDream AI, a friendly, expert transformation coach. "
                "Provide clear, step-by-step guidance and ask clarifying questions when needed."
            ),
        }
    ]

    # Inject memory into payload
    for entry in history:
        if entry.startswith("🧑:"):
            role = "user"
            content = entry[3:].strip()
        elif entry.startswith("🤖:"):
            role = "assistant"
            content = entry[3:].strip()
        else:
            continue
        messages_payload.append({"role": role, "content": content})

    # Append current user message
    messages_payload.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_payload,
            temperature=0.7,
            max_tokens=500,
            stream=False
        )
        reply = response.choices[0].message.content

        # Update and save memory
        new_memory = history + [f"🧑: {message}", f"🤖: {reply}"]
        save_memory(user_id, new_memory)

        return reply

    except OpenAIError as oe:
        return f"[OpenAI Error] {oe}"
    except Exception as e:
        return f"[Error] {e}"
