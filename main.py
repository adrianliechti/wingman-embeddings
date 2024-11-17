import os
import io
import base64
import uvicorn
import torch

from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import List, Optional

from PIL import Image
from transformers import AutoModel

model = AutoModel.from_pretrained(
    os.getenv("MODEL", "jinaai/jina-embeddings-v3"),
    trust_remote_code=True,
)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

model.eval()

encode_text = getattr(model, "encode_text", getattr(model, "encode", None))
encode_image = getattr(model, "encode_image", None)

app = FastAPI(
    title="LLM Platform Embeddings"
)

class InputItem(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None

class EmbeddingRequest(BaseModel):
    input: List[InputItem]

    @field_validator('input', mode='before')
    def parse_input(cls, v):
        if isinstance(v, str):
            v = [v]
        elif not isinstance(v, list):
            raise ValueError('Input must be a string or a list')
        
        input = []

        for item in v:
            if isinstance(item, str):
                input.append({'text': item})
            elif isinstance(item, dict):
                input.append(item)
            else:
                raise ValueError('Each item in input must be a string or a dictionary')
            
        return input

@app.post("/embeddings")
@app.post("/v1/embeddings")
def embed(request: EmbeddingRequest):
    input = request.input

    data = []
    
    for i, input in enumerate(input):
        if input.text:
            if encode_text is None:
                raise ValueError('Model does not have a method to encode text')
            
            embedding = encode_text(input.text)

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })
            
        if input.image:
            if encode_image == None:
                raise ValueError('Model does not have a method to encode image')
            
            if input.image.startswith("http://") or input.image.startswith("https://"):
                embedding = encode_image(input.image)
            else:
                image_data = base64.b64decode(input.image)

                image = Image.open(io.BytesIO(image_data))
                image = image.convert("RGB")

                embedding = encode_image(image)
            
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })
    
    return {
        "object": "list",
        "model": model.name_or_path,
        "data": data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)