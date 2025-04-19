from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Example input model
class MyRequest(BaseModel):
    text: str

@app.post("/process")
async def process(request: MyRequest):
    input_text = request.text
    output = f"Received and processed: {input_text}"
    return {"result": output}

@app.get("/")
async def root():
    return {"message": "Welcome to TextImages API!"}
