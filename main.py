import os
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAIError
from chat_agent import load_memory, save_memory, get_chat_response
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

def memory_file_path(user_id: str) -> str:
    return os.path.join(MEMORY_DIR, f"{user_id}.json")

# ─── Simple Chat Endpoint (non-streaming) ───────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_id   = data.get("user_id")
    message   = data.get("message", "")
    image_url = data.get("image_url")

    if not user_id or (not message and not image_url):
        return jsonify({"error": "Missing user_id or message/image_url"}), 400

    logger.info(f"[chat] user_id={user_id!r}, message={message!r}, image_url={image_url!r}")

    # Build memory entry for image or text
    memory = load_memory(user_id)
    if image_url and not message:
        memory.append({"role": "user", "content": f"[sent image: {image_url}]"})
    elif message:
        memory.append({"role": "user", "content": message})

    # Get response (vision-enabled)
    reply = get_chat_response(message, user_id, image_url)

    memory.append({"role": "assistant", "content": reply})
    save_memory(user_id, memory)

    return jsonify({"response": reply})

# ─── Streaming Chat Endpoint (SSE) ─────────────────────────────────────────────
@app.route("/chat/stream")
def chat_stream():
    user_id   = request.args.get("user_id")
    message   = request.args.get("message", "")
    image_url = request.args.get("image_url")
    if not user_id or (not message and not image_url):
        return ("Missing user_id or message/image_url", 400)

    logger.info(f"[chat/stream] user_id={user_id!r}, message={message!r}, image_url={image_url!r}")

    # Load history (last 20)
    history = load_memory(user_id)[-20:]

    # Build system + memory payload
    payload = [
        {
            "role": "system",
            "content": (
                "You are DayDream AI, a friendly, expert transformation coach. "
                "You can see and reason about images when provided. "
                "Respond with clear, step-by-step guidance and ask questions as needed."
            ),
        }
    ]
    for entry in history:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            payload.append(entry)

    # Embed image chunk if present
    if image_url:
        payload.append(
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": "Here is an image for you to analyze:"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        )

    # Append user text if present
    if message:
        payload.append({"role": "user", "content": message})

    def event_stream():
        full_reply = ""
        try:
            stream = get_chat_response.__globals__["client"].chat.completions.create(
                model="gpt-4-vision-preview",
                messages=payload,
                temperature=0.7,
                stream=True
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

        # Save history with image or text entries
        if image_url and not message:
            history.append({"role":"user","content":f"[sent image: {image_url}]"})
        elif message:
            history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":full_reply})
        save_memory(user_id, history)

        yield "event: done\ndata: \n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream"
    )

# ─── Image Generation Endpoint ─────────────────────────────────────────────────
@app.route("/image", methods=["POST"])
def generate_image():
    data     = request.get_json() or {}
    prompt   = data.get("prompt")
    user_id  = data.get("user_id")
    identity_image_url = data.get("identity_image_url")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        url = generate_image_from_prompt(prompt, identity_image_url)
        if url:
            return jsonify({"imageUrl": url})
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
