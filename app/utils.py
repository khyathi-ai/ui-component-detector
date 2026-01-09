import base64
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; UIComponentDetector/1.0)"
}

def load_image_from_input(image_input: str) -> bytes:
    """
    Accepts either a base64 string or an image URL
    Returns raw image bytes
    """
    if image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(
            image_input,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.content

    try:
        return base64.b64decode(image_input)
    except Exception:
        raise ValueError("Invalid image input. Must be a valid URL or base64 string.")
