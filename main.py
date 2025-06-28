import os
import json
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAIError
from chat_agent import load_memory, save_memory, get_chat_response, client
from image_generator import generate_image_from_prompt

app = Flask(__name__)
CORS(app)

MEMORY_DIR = "chat_memories"
os.makedirs(MEMORY_DIR, exist_ok=True)

def memory_file_path(user_id):
    return os.path.join(MEMORY_DIR, f"{user_id}.json")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("user_id")
    message = data.get("message")

    if not user_id or not message:
        return jsonify({"error": "Missing user_id or message"}), 400

    memory = load_memory(user_id)
    memory.append(f"🧑: {message}")

    response = get_chat_response(message, user_id)
    memory.append(f"🤖: {response}")

    save_memory(user_id, memory)
    return jsonify({"response": response})

@app.route("/chat/stream")
def chat_stream():
    # SSE streaming endpoint (GET only)
    user_id = request.args.get("user_id")
    message = request.args.get("message")
    if not user_id or not message:
        return ("Missing user_id or message", 400)

    # Build messages payload: system prompt, memory, user message
    history = load_memory(user_id)[-20:]
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
        if entry.startswith("🧑:"):
            payload.append({"role": "user", "content": entry[3:].strip()})
        elif entry.startswith("🤖:"):
            payload.append({"role": "assistant", "content": entry[3:].strip()})
    payload.append({"role": "user", "content": message})

    def event_stream():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model="gpt-4o", messages=payload, temperature=0.7, stream=True
            )
            for chunk in stream:
                # Extract content from ChoiceDelta object
                delta = getattr(chunk.choices[0].delta, "content", "") or ""
                full_reply += delta
                yield f"data: {delta}

"
        # yield f"data: {delta}\n\n"
        except OpenAIError as oe:
            yield f"event: error\ndata: [OpenAIError] {str(oe)}\n\n"
            return
        except Exception as e:
            yield f"event: error\ndata: [Error] {str(e)}\n\n"
            return

        # Save the completed conversation
        history.append(f"🧑: {message}")
        history.append(f"🤖: {full_reply}")
        save_memory(user_id, history)
        yield "event: done\ndata: \n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route("/image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt")
    user_id = data.get("user_id")
    identity_image_url = data.get("identity_image_url")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        if identity_image_url:
            url = generate_image_from_prompt(prompt, identity_image_url)
        else:
            url = generate_image_from_prompt(prompt)
        if url:
            return jsonify({"imageUrl": url})
        else:
            return jsonify({"error": "Image generation failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memory", methods=["GET", "POST"])
def memory():
    if request.method == "POST":
        data = request.get_json()
        user_id = data.get("user_id")
        messages = data.get("messages", [])
        save_memory(user_id, messages)
        return jsonify({"status": "saved", "count": len(messages)})

    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    messages = load_memory(user_id)
    return jsonify({"messages": messages})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
