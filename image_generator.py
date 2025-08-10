import os
import time
import base64
import logging
import requests
from typing import Optional
from collections import defaultdict
from openai import OpenAI, OpenAIError, BadRequestError, APIError, RateLimitError

print("[DEBUG] Using image_generator.py from:", __file__)

# ---- Logging ----
logger = logging.getLogger(__name__)

# ---- Clients / Keys ----
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SEGMIND_API_KEY = os.getenv("SEGMIND_API_KEY")
GETIMG_API_KEY = os.getenv("GETIMG_API_KEY")

# ---- Rate Limiting ----
last_request_time = defaultdict(float)
RATE_LIMIT_SECONDS = 10  # seconds between requests per user/session


# ---- Helpers ----
def _ensure_https_or_data_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("http://"):
        return "https://" + s[len("http://") :]
    if s.startswith("https://"):
        return s
    if s.startswith("data:image/"):
        return s
    # Heuristic for base64 payloads (e.g., Segmind "image")
    if len(s) > 100 and all(c.isalnum() or c in "+/=\n\r" for c in s[:120]):
        return "data:image/png;base64," + s.replace("\n", "")
    return s


def _url_to_base64(url: str) -> str:
    """Download an image and return base64-encoded content."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


# ---- Public Entrypoint ----
def generate_image_from_prompt(
    prompt: str,
    identity_image_url: Optional[str] = None,
    user_id: str = "global",
    high_quality: bool = False,
) -> str:
    """
    Routing rules:
      - If identity_image_url or high_quality: Segmind → Getimg → DALL·E.
      - Else: DALL·E → Segmind → Getimg.
    """
    now = time.time()
    if now - last_request_time[user_id] < RATE_LIMIT_SECONDS:
        wait = int(RATE_LIMIT_SECONDS - (now - last_request_time[user_id]))
        logger.info("[GEN IMG] Rate limited user=%s wait=%ss", user_id, wait)
        raise RuntimeError(f"Rate limit: Please wait {wait} seconds.")
    last_request_time[user_id] = now

    logger.info(
        "[GEN IMG] user=%s HQ=%s id_img=%s prompt=%r",
        user_id, high_quality, bool(identity_image_url), prompt,
    )

    # Prefer Segmind when identity/HQ is requested
    if identity_image_url or high_quality:
        logger.info("[Router] Prefer Segmind path first.")
        try:
            url = generate_with_segmind(prompt, identity_image_url)
            if url:
                logger.info("[Router] Segmind succeeded.")
                return url
            logger.warning("[Router] Segmind returned no URL; trying Getimg.")
        except Exception as e:
            logger.warning("[Router] Segmind error: %s. Trying Getimg.", e)

        try:
            url = generate_with_getimg(prompt, identity_image_url)
            if url:
                logger.info("[Router] Getimg succeeded.")
                return url
            logger.warning("[Router] Getimg returned no URL; trying DALL·E.")
        except Exception as e:
            logger.warning("[Router] Getimg error: %s. Trying DALL·E.", e)

        return generate_with_dalle(prompt)

    # Default: try DALL·E first, then Segmind, then Getimg
    logger.info("[Router] Prefer DALL·E path first.")
    try:
        return generate_with_dalle(prompt)
    except BadRequestError as e:
        logger.warning("[Router] DALL·E policy/bad request: %s. Trying Segmind.", e)
    except (RateLimitError, APIError, OpenAIError) as e:
        logger.warning("[Router] DALL·E API error: %s. Trying Segmind.", e)
    except Exception as e:
        logger.warning("[Router] DALL·E unexpected error: %s. Trying Segmind.", e)

    try:
        url = generate_with_segmind(prompt, identity_image_url)
        if url:
            logger.info("[Router] Segmind succeeded after DALL·E failure.")
            return url
    except Exception as e:
        logger.warning("[Router] Segmind error after DALL·E failure: %s. Trying Getimg.", e)

    url = generate_with_getimg(prompt, identity_image_url)
    if url:
        logger.info("[Router] Getimg succeeded after DALL·E/Segmind failure.")
        return url

    raise RuntimeError("All providers failed (DALL·E, Segmind, Getimg).")


# ---- OpenAI DALL·E ----
def generate_with_dalle(prompt: str) -> str:
    logger.info("[DALL·E] Called.")
    try:
        resp = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url = resp.data[0].url if resp and resp.data else None
        url = _ensure_https_or_data_url(url)
        if not url:
            raise RuntimeError("DALL·E returned no URL")
        logger.info("[DALL·E] Success.")
        return url
    except BadRequestError as e:
        logger.warning("[DALL·E] BadRequest (likely policy): %s", e)
        raise
    except (RateLimitError, APIError, OpenAIError) as e:
        logger.error("[DALL·E] API error: %s", e)
        raise
    except Exception as e:
        logger.exception("[DALL·E] Unexpected error")
        raise


# ---- Segmind InstantID ----
def generate_with_segmind(prompt: str, identity_image_url: Optional[str]) -> Optional[str]:
    logger.info("[Segmind] Called id_img=%s", bool(identity_image_url))
    try:
        if not SEGMIND_API_KEY:
            logger.error("[Segmind] Missing SEGMIND_API_KEY.")
            return None
        if not identity_image_url:
            logger.error("[Segmind] identity_image_url not provided.")
            return None

        headers = {
            "x-api-key": SEGMIND_API_KEY,   # header name per Segmind docs
            "Content-Type": "application/json",
        }

        face_b64 = _url_to_base64(identity_image_url)

        payload = {
            "prompt": prompt,
            "face_image": face_b64,          # base64 identity image
            "negative_prompt": "lowquality, badquality, sketches",
            "samples": 1,
            "num_inference_steps": 25,
            "guidance_scale": 6.5,
            "identity_strength": 0.8,
            "adapter_strength": 0.8,
            "enhance_face_region": True,
            "base64": True,                  # request base64 in JSON response
        }

        resp = requests.post(
            "https://api.segmind.com/v1/instantid",  # fixed endpoint
            json=payload,
            headers=headers,
            timeout=60,
        )

        logger.info("[Segmind] Status=%s", resp.status_code)
        if resp.status_code != 200:
            logger.error("[Segmind] Error %s: %s", resp.status_code, resp.text[:400])
            return None

        data = resp.json() or {}
        img_b64 = data.get("image") or (data.get("images") or [None])[0]
        if not img_b64:
            logger.error("[Segmind] No image in response.")
            return None
        return f"data:image/png;base64,{img_b64}"
    except Exception as e:
        logger.exception("[Segmind] Exception")
        return None


# ---- Getimg (ControlNet fallback) ----
def generate_with_getimg(prompt: str, identity_image_url: Optional[str]) -> Optional[str]:
    logger.info("[Getimg] Called id_img=%s", bool(identity_image_url))
    try:
        if not GETIMG_API_KEY:
            logger.error("[Getimg] Missing GETIMG_API_KEY.")
            return None
        if not identity_image_url:
            logger.error("[Getimg] identity_image_url not provided.")
            return None

        headers = {
            "Authorization": f"Bearer {GETIMG_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "image_url": identity_image_url,  # source image
            "controlnet": "pose",             # correct key (not control_type)
            "strength": 0.6,
            "guidance": 7,
            "steps": 25,
        }

        resp = requests.post(
            "https://api.getimg.ai/v1/stable-diffusion/controlnet",
            headers=headers,
            json=payload,
            timeout=60,
        )

        logger.info("[Getimg] Status=%s", resp.status_code)
        if resp.status_code != 200:
            logger.error("[Getimg] Error %s: %s", resp.status_code, resp.text[:400])
            return None

        data = resp.json() or {}
        url = data.get("image_url") or data.get("url")
        final = _ensure_https_or_data_url(url)
        logger.info("[Getimg] Final URL present=%s", bool(final))
        return final
    except Exception as e:
        logger.exception("[Getimg] Exception")
        return None
