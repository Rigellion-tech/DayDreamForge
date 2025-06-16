# image_generator.py

import os
from openai import OpenAI
from openai import OpenAIError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_image_from_prompt(prompt):
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
        return f"[Error] {e}"
