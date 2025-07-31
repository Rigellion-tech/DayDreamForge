import os
import time
import requests
from openai import OpenAI, OpenAIError
from collections import defaultdict

# Initialize clients and keys
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SEGMIND_API_KEY = os.getenv("SEGMIND_API_KEY")
GETIMG_API_KEY = os.getenv("GETIMG_API_KEY")

# ─── Rate Limiting ─────────────────────────────────────────────
last_request_time = defaultdict(float)
RATE_LIMIT_SECONDS = 10  # seconds between requests per user/session

def generate_image_from_prompt(prompt, identity_image_url=None, user_id="global", high_quality=False):
    now = time.time()
    if now - last_request_time[user_id] < RATE_LIMIT_SECONDS:
        raise RuntimeError(
            f"Rate limit: Please wait {int(RATE_LIMIT_SECONDS - (now - last_request_time[user_id]))} seconds."
        )
    last_request_time[user_id] = now

    try:
        # Priority: Segmind → Getimg → DALL·E
        if high_quality or identity_image_url:
            image = generate_with_segmind(prompt, identity_image_url)
            if image:
                return image
            image = generate_with_getimg(prompt, identity_image_url)
            if image:
                return image

        return generate_with_dalle(prompt)

    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")


# ─── OpenAI DALL·E ──────────────────────────────────────────────
def generate_with_dalle(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unhandled Error in DALL·E: {e}")


# ─── Segmind InstantID ──────────────────────────────────────────
def generate_with_segmind(prompt, identity_image_url):
    try:
        headers = {
            "X-API-KEY": SEGMIND_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "identity_image": identity_image_url,
            "model": "instantid",
            "enhance_prompt": True,
            "scheduler": "DPM++ SDE Karras",
            "num_inference_steps": 25,
            "guidance_scale": 6.5
        }
        response = requests.post(
            "https://api.segmind.com/v1/sd/instantid",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("image")
        else:
            print(f"[Segmind Error] {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"[Segmind Call Failed] {e}")
        return None


# ─── Getimg (ControlNet Fallback) ──────────────────────────────
def generate_with_getimg(prompt, identity_image_url):
    try:
        headers = {
            "Authorization": f"Bearer {GETIMG_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "controlnet",
            "prompt": prompt,
            "image_url": identity_image_url,
            "control_type": "pose",
            "guidance": 7,
            "strength": 0.6,
            "steps": 25
        }

        response = requests.post(
            "https://api.getimg.ai/v1/stable-diffusion/controlnet",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("image_url")
        else:
            print(f"[Getimg Error] {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"[Getimg Call Failed] {e}")
        return None
