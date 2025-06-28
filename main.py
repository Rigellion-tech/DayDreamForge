from flask import Flask, request, jsonify
from flask_cors import CORS
from chat_agent import get_chat_response
from image_generator import generate_image_from_prompt
import os
import json

app = Flask(__name__)
CORS(app)

MEMORY_DIR = "chat_memories"
os.makedirs(MEMORY_DIR, exist_ok=True)

def memory_file_path(user_id):
    return os.path.join(MEMORY_DIR, f"{user_id}.json")

def load_memory(user_id):
    try:
        with open(memory_file_path(user_id), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_memory(user_id, messages):
    with open(memory_file_path(user_id), "w") as f:
        json.dump(messages, f)

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

# Updated route and response key for image generation
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
            image_url = generate_image_from_prompt(prompt, identity_image_url)
        else:
            image_url = generate_image_from_prompt(prompt)

        if image_url:
            return jsonify({"imageUrl": image_url})  # match frontend expectation
        else:
            return jsonify({"error": "Image generation failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memory", methods=["GET", "POST"])
def memory():
    data = request.get_json() if request.method == "POST" else request.args
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    if request.method == "POST":
        messages = data.get("messages", [])
        save_memory(user_id, messages)
        return jsonify({"status": "saved", "count": len(messages)})

    messages = load_memory(user_id)
    return jsonify({"messages": messages})
@app.route("/chat/stream")
def chat_stream():
    # read user_id and message from query params since EventSource uses GET
    user_id = request.args.get("user_id")
    message = request.args.get("message")
    if not user_id or not message:
        return ("Missing user_id or message", 400)

    # Re-use your chat_agent logic to build the messages_payload
    # (system prompt + memory + user message)
    messages_payload = build_payload(message, user_id)

    def generate():
        try:
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=messages_payload,
                temperature=0.7,
                stream=True,
            )
            full = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.get("content")
                if delta:
                    full += delta
                    yield f"data: {delta}\n\n"
            # once complete, save the full assistant reply into memory
            save_reply_to_memory(user_id, message, full)
            yield "event: done\ndata: \n\n"
        except OpenAIError as oe:
            yield f"event: error\ndata: [OpenAIError] {str(oe)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: [Error] {str(e)}\n\n"

    # CORS is already enabled for all routes via CORS(app)
    return Response(
        stream_with_context(generate()), 
        mimetype="text/event-stream"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
