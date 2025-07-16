import os
import json
import re
import random
import string
import datetime
import logging
from flask import Flask, request, jsonify, Response, stream_with_context, make_response
from flask_cors import CORS
from openai import OpenAIError
from auth_email import send_login_code
from chat_agent import load_memory, save_memory, get_chat_response
from image_generator import generate_image_from_prompt

# ─── Logging Setup ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Flask App Setup ──────────────────────────────────────────────
app = Flask(__name__)

# ✅ PATCHED CORS
CORS(
    app,
    origins=[
        "https://daydreamforge.vercel.app",
    ],
    supports_credentials=True,
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)

# ─── Directories ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "chat_memories")
AUTH_CODES_DIR = os.path.join(BASE_DIR, "auth_codes")
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(AUTH_CODES_DIR, exist_ok=True)

# ─── Auth Code Helpers ────────────────────────────────────────────

def generate_code(length=6):
    return ''.join(random.choices(string.digits, k=length))

def save_code(email, code):
    path = os.path.join(AUTH_CODES_DIR, f"{email}.json")
    data = {
        "code": code,
        "expires_at": (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()
    }
    with open(path, "w") as f:
        json.dump(data, f)

def verify_code(email, code):
    path = os.path.join(AUTH_CODES_DIR, f"{email}.json")
    if not os.path.exists(path):
        return False, "No code sent to this email."

    with open(path, "r") as f:
        stored = json.load(f)

    expires = datetime.datetime.fromisoformat(stored["expires_at"])
    if datetime.datetime.utcnow() > expires:
        return False, "Code expired."

    if stored["code"] != code:
        return False, "Invalid code."

    os.remove(path)
    return True, None

# ─── Auth Endpoints ───────────────────────────────────────────────

@app.route("/auth/request_code", methods=["POST"])
def request_code():
    data = request.get_json() or {}
    email = data.get("email")

    logger.info(f"Received auth request_code for email: {email}")

    if not email:
        return jsonify({"error": "Email is required"}), 400

    code = generate_code()
    save_code(email, code)
    send_login_code(email, code)

    return jsonify({"success": True})

@app.route("/auth/verify_code", methods=["POST"])
def verify_auth_code():
    data = request.get_json() or {}
    email = data.get("email")
    code = data.get("code")

    if not email or not code:
        return jsonify({"error": "Missing email or code"}), 400

    ok, error_msg = verify_code(email, code)
    if not ok:
        return jsonify({"error": error_msg}), 400

    # Success — use email as user_id
    user_id = email

    response = make_response(jsonify({"success": True, "user_id": user_id}))
    response.set_cookie(
        "user_id",
        user_id,
        max_age=60 * 60 * 24 * 365,  # one year
        path="/",
        secure=True,
        httponly=True,
        samesite="Lax"
    )
    return response

# ─── Chat Endpoints ───────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_id = data.get("user_id")
    message = data.get("message", "")
    image_url = data.get("image_url")

    if not user_id or (not message and not image_url):
        return jsonify({"error": "Missing user_id or message/image_url"}), 400

    logger.info(f"[chat] user_id={user_id!r}, message={message!r}, image_url={image_url!r}")

    memory = load_memory(user_id)
    if image_url and not message:
        memory.append({"role": "user", "content": f"[sent image: {image_url}]"})
    elif message:
        memory.append({"role": "user", "content": message})

    reply = get_chat_response(message, user_id, image_url)

    memory.append({"role": "assistant", "content": reply})
    save_memory(user_id, memory)

    return jsonify({"response": reply})

# ─── Streaming Chat Endpoint ─────────────────────────────────────

@app.route("/chat/stream", methods=["GET", "POST", "OPTIONS"])
def chat_stream():
    if request.method == "OPTIONS":
        return "", 200

    if request.method == "POST":
        data = request.get_json() or {}
        user_id = data.get("user_id")
        image_url = data.get("image_url")
        messages = data.get("messages", [])
    else:
        user_id = request.args.get("user_id")
        image_url = request.args.get("image_url")
        message = request.args.get("message", "")
        messages = []
        if message:
            messages.append({"role": "user", "content": message})

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    logger.info(f"[chat/stream] user_id={user_id!r}, image_url={image_url!r}")

    history = load_memory(user_id)[-20:]

    payload = [
        {
            "role": "system",
            "content": (
                "You are DayDream AI, a friendly, expert transformation coach. "
                "You can see and reason about images when provided. "
                "Respond with clear, step-by-step guidance and ask questions as needed."
            )
        }
    ]

    for entry in history:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            payload.append(entry)

    payload += messages

    if image_url:
        payload.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is an image for you to analyze:"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        })

    def event_stream():
        full_reply = ""
        try:
            stream = get_chat_response.__globals__["client"].chat.completions.create(
                model="gpt-4o",
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
            yield f"data: [OpenAIError] {oe}\n\n"
            yield "event: done\ndata: \n\n"
            return

        except Exception as e:
            logger.exception("Error in stream")
            yield f"data: [Error] {e}\n\n"
            yield "event: done\ndata: \n\n"
            return

        if image_url and not messages:
            history.append({"role": "user", "content": f"[sent image: {image_url}]"})
        elif messages:
            history.extend(messages)

        history.append({"role": "assistant", "content": full_reply})
        save_memory(user_id, history)

        yield "event: done\ndata: \n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream"
    )

# ─── Image Generation ───────────────────────────────────────────

@app.route("/image", methods=["POST"])
def generate_image():
    data = request.get_json() or {}
    prompt = data.get("prompt")
    user_id = data.get("user_id")
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

# ─── Memory Inspection & Save ──────────────────────────────────

@app.route("/memory", methods=["GET", "POST"])
def memory():
    if request.method == "POST":
        data = request.get_json() or {}
        user_id = data.get("user_id")
        messages = data.get("messages", [])
        save_memory(user_id, messages)
        return jsonify({"status": "saved", "count": len(messages)})

    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    return jsonify({"messages": load_memory(user_id)})

# ─── Entrypoint ────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
