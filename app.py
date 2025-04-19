from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from diffusers import DiffusionPipeline
import random
import sys
import os

app = FastAPI()

# Load model ONCE at startup (saves memory)
pipe = None

@app.on_event("startup")
async def load_model():
    global pipe
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()  # Reduces memory usage
    pipe.set_progress_bar_config(disable=True)  # Hide progress bars

class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    seed: int | None = None

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    if not pipe:
        raise HTTPException(status_code=500, detail="Model not loaded yet")

    seed = request.seed if request.seed else random.randint(0, sys.maxsize)
    
    try:
        image = pipe(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

        # Save temporarily (Render has ephemeral storage)
        output_path = "output.jpg"
        image.save(output_path)
        
        return FileResponse(output_path, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "POST /generate-image with a 'prompt' to generate images."}