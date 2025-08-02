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
is_prod = app.config.get("ENV") == "production"

# ─── CORS Configuration ───────────────────────────────────────────
CORS(
    app,
    origins=[
        "https://daydreamforge.com",
        "https://www.daydreamforge.com",
        "https://daydreamforge.vercel.app",
        "http://localhost:3000",
        "https://daydreamforge.onrender.com",
    ],
    supports_credentials=True,
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)

# ─── Routes ───────────────────────────────────────────────────────

@app.route("/generate-image", methods=["POST"])
def generate_image():
    try:
        data = request.json
        prompt = data.get("prompt")
        identity_image_url = data.get("identity_image_url")  # Optional

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        logger.info(f"[IMAGE GENERATION] Prompt: {prompt}")
        if identity_image_url:
            logger.info(f"[IMAGE GENERATION] Identity Image Provided: {identity_image_url}")

        image_url = generate_image_from_prompt(prompt, identity_image_url)

        if image_url:
            return jsonify({"image_url": image_url}), 200
        else:
            return jsonify({"error": "Image generation failed"}), 500

    except Exception as e:
        logger.exception("[ERROR] Image generation failed")
        return jsonify({"error": str(e)}), 500

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "chat_memories")
AUTH_CODES_DIR = os.path.join(BASE_DIR, "auth_codes")

os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(AUTH_CODES_DIR, exist_ok=True)

# ─── Auth Code Helpers ───
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

# ─── Auth Endpoints ───
@app.before_request
def log_request():
    logger.info(f"{request.method} {request.path}")

@app.route("/auth/request_code", methods=["POST", "OPTIONS"])
def request_code():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json() or {}
    email = data.get("email")
    logger.info(f"Received auth request_code for email: {email}")
    if not email:
        return jsonify({"error": "Email is required"}), 400
    code = generate_code()
    save_code(email, code)
    try:
        logger.info(f"Sending login code {code} to {email}")
        send_login_code(email, code)
    except Exception as e:
        logger.exception("Failed to send login email")
        return jsonify({"error": "Failed to send login email."}), 500
    return jsonify({"success": True}), 200

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
    user_id = email
    response = make_response(jsonify({"success": True, "user_id": user_id}))
    cookie_args = {
        "path": "/",
        "httponly": False,
        "samesite": "Lax",
    }
    if is_prod:
        cookie_args.update({
            "max_age": 60 * 60 * 24 * 365,
            "secure": True,
            "domain": ".daydreamforge.com",
        })
    else:
        cookie_args["max_age"] = 60 * 60 * 24 * 365
    response.set_cookie("user_id", user_id, **cookie_args)
    return response

@app.route("/auth/logout", methods=["POST"])
def logout():
    response = make_response(jsonify({"success": True}))
    cookie_args = {
        "expires": 0,
        "path": "/",
        "httponly": False,
        "samesite": "Lax",
    }
    if is_prod:
        cookie_args.update({
            "secure": True,
            "domain": ".daydreamforge.com",
        })
    response.set_cookie("user_id", "", **cookie_args)
    return response

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

    # --- PATCHED LOGIC: Always call Segmind/Getimg when image is present ---
    if image_url:
        try:
            logger.info(f"[/chat] Routing to generate_image_from_prompt (Segmind/Getimg). prompt={message} | img={image_url}")
            img_url = generate_image_from_prompt(message or "transform this image", image_url)
            memory.append({"role": "user", "content": f"[prompt+image: {message or '[no message]'} + {image_url}]"})
            memory.append({"role": "assistant", "content": f"[Generated Image:]({img_url})"})
            save_memory(user_id, memory)
            return jsonify({"response": img_url, "imageUrl": img_url})
        except Exception as e:
            logger.exception("Image generation failed in /chat (Segmind/Getimg)")
            return jsonify({"error": str(e)}), 500

    # --- Fallback: Normal chat behavior for plain message only ---
    if message:
        memory.append({"role": "user", "content": message})
    reply = get_chat_response(message, user_id, image_url)
    memory.append({"role": "assistant", "content": reply})
    save_memory(user_id, memory)
    return jsonify({"response": reply})

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
        {"role": "system", "content": (
            "You are DayDream AI, a world-class transformation coach and motivator. "
            "You are wise, calm, and deeply supportive. "
            "You help users achieve personal growth and overcome obstacles, always responding with warmth and encouragement.\n\n"
            "Your style is thoughtful, Socratic, and inspiring—like a wise mentor who remembers previous conversations. "
            "Whenever possible, use insights from earlier messages to make your responses more personal and relevant.\n\n"
            "Be concise, practical, and optimistic. Ask deep, reflective questions and provide actionable, motivating advice. "
            "If the user shares an image, analyze and interpret it as a skilled coach would. "
            "Never give up on the user—always believe in their ability to improve. "
            "If you need to challenge them, do so gently, with empathy and insight.\n\n"
            "Format your responses in clear paragraphs, and make every message feel like a personal, supportive conversation."
        )}
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
                temperature=0.55,
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
