from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from local_client import generate_image_remote

app = FastAPI()

class ImgRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    seed: int = None

@app.post("/generate-image")
async def generate_image(req: ImgRequest):
    try:
        img_path = generate_image_remote(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            seed=req.seed,
        )
        return {"status": "success", "filepath": str(img_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with:
# uvicorn main:app --reload
