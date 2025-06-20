from flask import Flask, request, jsonify
from flask_cors import CORS
from chat_agent import get_chat_response
from image_generator import generate_image_from_prompt

app = Flask(__name__)
CORS(app)  # Allow all cross-origin requests
chat_memory = []

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = get_chat_response(user_input)
    return jsonify({"response": response})

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt")
    identity_image_url = data.get("identity_image_url")  # Optional
    image_url = generate_image_from_prompt(prompt, identity_image_url)
    return jsonify({"image_url": image_url})


@app.route("/memory", methods=["GET", "POST"])
def memory():
    global chat_memory
    if request.method == "POST":
        data = request.get_json()
        chat_memory = data.get("messages", [])
        return jsonify({"status": "saved", "count": len(chat_memory)})
    return jsonify({"messages": chat_memory})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
