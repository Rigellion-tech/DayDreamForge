# chat_agent.py
import os
from openai import OpenAI
from openai import OpenAIError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_chat_response(message):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a motivational transformation coach."},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"[OpenAI Error] {e}"
    except Exception as e:
        return f"[Error] {e}"
