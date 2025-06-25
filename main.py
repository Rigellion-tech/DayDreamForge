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

    response = get_chat_response(message)
    memory.append(f"🤖: {response}")

    save_memory(user_id, memory)

    return jsonify({"response": response})

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt")
    user_id = data.get("user_id")
    identity_image_url = data.get("identity_image_url")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        # Check if we should use InstantID
        if identity_image_url:
            image_url = generate_image_from_prompt(prompt, identity_image_url)
        else:
            image_url = generate_image_from_prompt(prompt)

        if image_url:
            return jsonify({"image_url": image_url})
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
