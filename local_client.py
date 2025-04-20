import requests, base64
from pathlib import Path

# Replace with your actual ngrok URL from Colab
NGROK_URL = "https://17d0-34-124-254-60.ngrok-free.app"
ENDPOINT = f"{NGROK_URL}/generate-image"

def generate_image_remote(prompt: str, width: int = 512, height: int = 512, seed: int = None) -> Path:
    payload = {"prompt": prompt, "width": width, "height": height}
    if seed is not None:
        payload["seed"] = seed

    # 1) Call the Colab endpoint
    resp = requests.post(ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # 2) Decode the base64 PNG
    img_b64 = data["image"]
    img_bytes = base64.b64decode(img_b64)

    # 3) Save locally
    out = Path("generated_images")
    out.mkdir(exist_ok=True)
    fname = f"{data['seed']}.png"
    fpath = out / fname
    fpath.write_bytes(img_bytes)

    return fpath

if __name__ == "__main__":
    path = generate_image_remote("a futuristic city at dawn", seed=42)
    print(f"Saved image to {path}")
