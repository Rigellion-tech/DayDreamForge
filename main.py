from flask import Flask, request, jsonify
from chat_agent import get_chat_response
from image_generator import generate_image_from_prompt

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = get_chat_response(user_input)
    return jsonify({"response": response})

@app.route("/generate-image", methods=["POST"])
@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt")
    identity_image_url = data.get("identity_image_url")  # Optional
    image_url = generate_image_from_prompt(prompt, identity_image_url)
    return jsonify({"image_url": image_url})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
