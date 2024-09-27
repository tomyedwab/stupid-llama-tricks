import pydantic

# Example config:
#    "name": "Phi3-mini-1.0",
#    "model": "./models/Phi-3-mini-4k-instruct-q4.gguf",
#    "context_size": 4096,
#    "temperature": 1.0,
#    "batch_size": 512,
#    "batch_max_tokens": 2048,

class ModelConfig(pydantic.BaseModel):
    name: str
    model_filename: str
    context_size: int
    temperature: float
    batch_size: int
    batch_max_tokens: int
    