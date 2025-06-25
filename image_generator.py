import os
import time
import requests
from openai import OpenAI, OpenAIError
from collections import defaultdict

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SEGMIND_API_KEY = os.getenv("SEGMIND_API_KEY")

# Rate limiting (basic in-memory for testing)
last_request_time = defaultdict(float)
RATE_LIMIT_SECONDS = 10  # seconds between requests per user/session (adjust as needed)

def generate_image_from_prompt(prompt, identity_image_url=None, user_id="global"):
    now = time.time()
    if now - last_request_time[user_id] < RATE_LIMIT_SECONDS:
        return f"[Rate Limit] Please wait {int(RATE_LIMIT_SECONDS - (now - last_request_time[user_id]))} seconds."

    last_request_time[user_id] = now

    try:
        if identity_image_url:
            return generate_with_segmind(prompt, identity_image_url)
        else:
            return generate_with_dalle(prompt)
    except Exception as e:
        return f"[Image Generation Error] {e}"

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
        return f"[OpenAI Error] {e}"
    except Exception as e:
        return f"[Unhandled Error in DALL·E] {e}"

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
        response = requests.post("https://api.segmind.com/v1/sd/instantid", json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data.get("image")
        else:
            print(f"[Segmind Error] {response.status_code}: {response.text}")
            return generate_with_dalle(prompt)

    except Exception as e:
        print(f"[Segmind Call Failed] {e}")
        return generate_with_dalle(prompt)
