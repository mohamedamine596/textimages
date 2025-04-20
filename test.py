import requests
import base64
from pathlib import Path

# Use the same URL as in your working local_client.py
NGROK_URL = "https://17d0-34-124-254-60.ngrok-free.app"
ENDPOINT = f"{NGROK_URL}/generate-image"

payload = {
    "prompt": "A beautiful landscape with mountains and a river make it realistic",
    "width": 512,
    "height": 512
}

try:
    # Make request with proper error handling
    res = requests.post(ENDPOINT, json=payload)
    res.raise_for_status()  # This will raise an exception for HTTP errors
    
    # Print status for debugging
    print(f"Status code: {res.status_code}")
    
    data = res.json()
    
    # Print the response data to see its structure
    print("Response data structure:")
    print(data)
    
    # Create output directory
    out = Path("generated_images")
    out.mkdir(exist_ok=True)
    
    # Try to get the image data with different possible keys
    if "image" in data:
        img_key = "image"
    elif "image_base64" in data:
        img_key = "image_base64"
    else:
        raise KeyError(f"No image key found in response. Available keys: {list(data.keys())}")
    
    img_data = base64.b64decode(data[img_key])
    
    # Save with seed as filename if available, otherwise use timestamp
    if "seed" in data:
        fname = f"{data['seed']}.png"
    else:
        import time
        fname = f"image_{int(time.time())}.png"
    
    fpath = out / fname
    fpath.write_bytes(img_data)
    
    print(f"Image saved as {fpath}")
    
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
    if hasattr(e, 'response') and e.response:
        print(f"Response content: {e.response.text[:500]}...")
except KeyError as e:
    print(f"Key error: {e}")
except ValueError as e:
    print(f"JSON decode error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()