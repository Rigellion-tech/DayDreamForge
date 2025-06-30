import os
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAIError
from chat_agent import load_memory, save_memory, get_chat_response, client
from image_generator import generate_image_from_prompt

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Flask App Setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─── Memory Directory (absolute path) ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "chat_memories")
os.makedirs(MEMORY_DIR, exist_ok=True)

def memory_file_path(user_id):
    return os.path.join(MEMORY_DIR, f"{user_id}.json")

# ─── Simple Chat Endpoint (non-streaming) ───────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_id = data.get("user_id")
    message = data.get("message")

    if not user_id or not message:
        return jsonify({"error": "Missing user_id or message"}), 400

    logger.info(f"[chat] user_id={user_id} message={message!r}")
    # load, append, get response, append, save
    memory = load_memory(user_id)
    memory.append(f"🧑: {message}")

    reply = get_chat_response(message, user_id)

    memory.append(f"🤖: {reply}")
    save_memory(user_id, memory)

    return jsonify({"response": reply})

# ─── Streaming Chat Endpoint (SSE) ─────────────────────────────────────────────
@app.route("/chat/stream")
def chat_stream():
    user_id = request.args.get("user_id")
    message = request.args.get("message")
    if not user_id or not message:
        return ("Missing user_id or message", 400)

    logger.info(f"[chat/stream] user_id={user_id} message={message!r}")

    # Load last 20 memory entries
    history = load_memory(user_id)[-20:]

    # Build the OpenAI payload
    payload = [
        {
            "role": "system",
            "content": (
                "You are DayDream AI, a friendly, expert transformation coach. "
                "Provide clear, step-by-step guidance and ask clarifying questions when needed."
            ),
        }
    ]
    for entry in history:
        # If someone has already migrated to dicts:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            payload.append(entry)

        # Otherwise, fall back to your emoji-string format:
        elif isinstance(entry, str):
            if entry.startswith("🧑:"):
                payload.append({"role": "user",      "content": entry[3:].strip()})
            elif entry.startswith("🤖:"):
                payload.append({"role": "assistant", "content": entry[3:].strip()})

    # Finally, our new user message
    payload.append({"role": "user", "content": message})

    def event_stream():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model="gpt-4o", messages=payload, temperature=0.7, stream=True
            )
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", "") or ""
                full_reply += delta
                yield f"data: {delta}\n\n"

        except OpenAIError as oe:
            logger.exception("OpenAIError in stream")
            yield f"event: error\ndata: [OpenAIError] {oe}\n\n"
            return

        except Exception as e:
            logger.exception("Error in stream")
            yield f"event: error\ndata: [Error] {e}\n\n"
            return

        # Save the conversation once complete
        history.append(f"🧑: {message}")
        history.append(f"🤖: {full_reply}")
        save_memory(user_id, history)

        yield "event: done\ndata: \n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream"
    )

# ─── Image Generation Endpoint ─────────────────────────────────────────────────
@app.route("/image", methods=["POST"])
def generate_image():
    data = request.get_json() or {}
    prompt             = data.get("prompt")
    user_id            = data.get("user_id")
    identity_image_url = data.get("identity_image_url")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        url = generate_image_from_prompt(prompt, identity_image_url)
        if url:
            return jsonify({"imageUrl": url})
        else:
            return jsonify({"error": "Image generation failed"}), 500

    except Exception as e:
        logger.exception("Error in /image")
        return jsonify({"error": str(e)}), 500

# ─── Memory Inspection & Save Endpoint ─────────────────────────────────────────
@app.route("/memory", methods=["GET", "POST"])
def memory():
    if request.method == "POST":
        data     = request.get_json() or {}
        user_id  = data.get("user_id")
        messages = data.get("messages", [])
        save_memory(user_id, messages)
        return jsonify({"status": "saved", "count": len(messages)})

    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    return jsonify({"messages": load_memory(user_id)})

# ─── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
