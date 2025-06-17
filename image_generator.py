import os
import requests

SEGMIND_API_KEY = os.getenv("SEGMIND_API_KEY")

def generate_image_from_prompt(prompt, identity_image_url=None):
    try:
        if identity_image_url:
            return generate_with_segmind(prompt, identity_image_url)
        else:
            return generate_with_dalle(prompt)
    except Exception as e:
        return f"[Error] {e}"

def generate_with_dalle(prompt):
    from openai import OpenAI, OpenAIError
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
            return response.json()["image"]
        else:
            return f"[Segmind Error] {response.status_code}: {response.text}"
    except Exception as e:
        return f"[Segmind Call Failed] {e}"
