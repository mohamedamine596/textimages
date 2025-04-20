import requests
import base64

payload = {
    "prompt": "A bold man",
    "width": 512,
    "height": 512
}

res = requests.post("https://da1d-34-87-159-177.ngrok-free.app/generate-image", json=payload)
data = res.json()

# Save the image
img_data = base64.b64decode(data["image_base64"])
with open("test_image.png", "wb") as f:
    f.write(img_data)

print(f"Image saved as test_image.png | seed: {data['seed']}")
