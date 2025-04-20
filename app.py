from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import random
import sys
import torch
import io
import base64
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Add this import
from diffusers import DiffusionPipeline
import os

app = FastAPI()

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the images directory to serve static files
os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")

# Initialize the model once at startup
use_refiner = False
device = "cpu"  # Use CPU as default for limited VRAM systems

# This will be lazy-loaded on the first request
pipe = None
refiner = None

def load_models():
    global pipe, refiner
    if pipe is None:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,  # Use float32 for CPU
            use_safetensors=True,
        )
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        
        if use_refiner:
            refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=pipe.text_encoder_2,
                vae=pipe.vae,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )

class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    use_refiner: bool = False
    seed: int = None

class MyRequest(BaseModel):
    text: str

@app.post("/process")
async def process(request: MyRequest):
    input_text = request.text
    output = f"Received and processed: {input_text}"
    return {"result": output}

@app.post("/generate-image")
async def generate_image(request: ImageRequest, background_tasks: BackgroundTasks):
    global pipe, refiner, use_refiner
    
    # Load models if not already loaded
    if pipe is None:
        load_models()
    
    # Set seed if provided, otherwise generate random seed
    if request.seed is None:
        seed = random.randint(0, sys.maxsize)
    else:
        seed = request.seed
    
    use_refiner = request.use_refiner
    
    try:
        # Generate the image
        images = pipe(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            output_type="latent" if use_refiner else "pil",
            generator=torch.Generator().manual_seed(seed),
        ).images

        if use_refiner and refiner is not None:
            images = refiner(
                prompt=request.prompt,
                image=images,
            ).images

        # Convert PIL image to base64 string
        buffered = io.BytesIO()
        images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Save image in images folder (create if not exists)
        os.makedirs("images", exist_ok=True)
        filename = f"generated_{seed}.png"
        filepath = f"images/{filename}"
        images[0].save(filepath)
        
        # Return the URL path that can be used to access the image
        image_url = f"/images/{filename}"
        
        return {
            "image": img_str,
            "prompt": request.prompt,
            "seed": seed,
            "filename": filepath,
            "image_url": image_url  # Add this for direct access
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Welcome to TextImages API!"}

if __name__ == "__main__":
    import uvicorn
    # Run the API server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)