from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import time

from modules.model_services.schemas import ChatCompletionRequest
from modules.model_services.inferences import call_ollama

app = FastAPI()

@app.get("/status")
def get_api_status() -> bool:
    return True


# 4. Define the OpenAI Compatible Endpoint
@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    try:
        content = call_ollama(
            model=request.model, 
            messages=request.model_dump()["messages"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model inference failed: {str(e)}"
        )

    # 5. Format and return standard OpenAI response dictionary
    return {
        "id": "chatcmpl-kaggle-ocr",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # Placeholders for compatibility
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }