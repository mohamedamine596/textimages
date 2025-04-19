from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Example input model
class MyRequest(BaseModel):
    text: str

@app.post("/process")
async def process(request: MyRequest):
    input_text = request.text
    # 🔥 Here you call your model functions
    output = f"Received and processed: {input_text}"
    return {"result": output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
