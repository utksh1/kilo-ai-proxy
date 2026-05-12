from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import uuid
import json
import logging
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KiloProxy")

# Environment Variables for Cloud Hosting
API_KEY = os.getenv("PROXY_API_KEY", "abc")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "inclusionai/ring-2.6-1t:free")
PORT = int(os.getenv("PORT", 3005))

security = HTTPBearer()

app = FastAPI(
    title="Kilo AI Proxy (Unofficial)",
    description="An OpenAI-compatible gateway to Kilo Code's free AI models. Features automatic Machine ID rotation to bypass rate limits.",
    version="1.1.0",
    contact={
        "name": "Kilo Proxy Support",
        "url": "https://github.com/Kilo-Org/kilocode",
    }
)

KILO_GATEWAY_URL = "https://api.kilo.ai/api/gateway/chat/completions"
DEFAULT_USER_AGENT = "Kilo CLI"
DEFAULT_MODEL = "inclusionai/ring-2.6-1t:free"

# Full list of free models discovered in Kilo binary (Ordered by IQ & Parameters)
FREE_MODELS = [
    {"id": "inclusionai/ring-2.6-1t:free", "name": "🌌 Default: Ring 2.6 1T (Largest Model)", "owned_by": "inclusionai"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "name": "🐘 Heavyweight: Nemotron 3 Super 120B", "owned_by": "nvidia"},
    {"id": "openrouter/free", "name": "🏆 Smartest: OpenRouter Free (Best IQ)", "owned_by": "openrouter"},
    {"id": "stepfun/step-3.5-flash:free", "name": "⚡ Fastest Smart: Step 3.5 Flash", "owned_by": "stepfun"},
    {"id": "poolside/laguna-xs.2:free", "name": "🧠 Deep Logic: Laguna XS.2", "owned_by": "poolside"},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "name": "🦉 Reasoning: Nemotron 3 Owl Alpha", "owned_by": "nvidia"},
    {"id": "kilo-auto/free", "name": "Kilo Auto (Dynamic Router)", "owned_by": "kilo"},
    {"id": "x-ai/grok-code-fast-1:optimized:free", "name": "Grok Code Fast 1 (Free)", "owned_by": "xai"},
    {"id": "poolside/laguna-m.1:free", "name": "Laguna M.1 (Free)", "owned_by": "poolside"},
    {"id": "baidu/cobuddy:free", "name": "Baidu Cobuddy (Free)", "owned_by": "baidu"},
    {"id": "mimo-v2-flash", "name": "Xiaomi MiMo V2 Flash", "owned_by": "xiaomi"},
    {"id": "nova-2-lite-v1", "name": "Amazon Nova 2 Lite", "owned_by": "amazon"}
]

# Models for Swagger documentation and validation
class ChatMessage(BaseModel):
    role: str = Field(..., example="user")
    content: str = Field(..., example="Hello!")

class ChatRequest(BaseModel):
    model: Optional[str] = Field(DEFAULT_MODEL, example=DEFAULT_MODEL)
    messages: List[ChatMessage]
    stream: Optional[bool] = Field(False, example=False)
    temperature: Optional[float] = Field(0.7, example=0.7)
    max_tokens: Optional[int] = Field(None, example=1024)

@app.post("/v1/chat/completions", tags=["Chat"])
async def chat_completions(
    request_data: ChatRequest, 
    request: Request, 
    auth: HTTPAuthorizationCredentials = Security(security)
):
    # 0. Check API Key
    if auth.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key. Use 'abc'")

    # 1. Prepare the body for Kilo
    body = request_data.dict(exclude_none=True)

    # 2. Prepare headers
    # Priority: 1. Header from client, 2. Random UUID
    machine_id = request.headers.get("X-Kilo-Machine-Id") or str(uuid.uuid4())
    
    headers = {
        "Content-Type": "application/json",
        "X-KILOCODE-MACHINEID": machine_id,
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json"
    }

    # 3. Ensure we use a free model if not specified
    if "model" not in body or not body["model"]:
        body["model"] = "kilo-auto/free"
    
    logger.info(f"Forwarding request to Kilo (Model: {body['model']}, MachineID: {machine_id})")

    # 4. Proxy the request to Kilo Gateway
    is_streaming = body.get("stream", False)

    async def stream_generator():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", KILO_GATEWAY_URL, json=body, headers=headers) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    logger.error(f"Kilo API Error: {response.status_code} - {error_detail.decode()}")
                    yield f"data: {json.dumps({'error': 'Kilo API error', 'status': response.status_code})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"

    if is_streaming:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(KILO_GATEWAY_URL, json=body, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Kilo API Error: {e.response.status_code} - {e.response.text}")
                return JSONResponse(status_code=e.response.status_code, content=e.response.json())
            except Exception as e:
                logger.error(f"Proxy Error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "online", "proxy": "Kilo AI Gateway", "auth": "enabled"}

@app.get("/v1/models", tags=["Models"])
async def list_models():
    """Returns the full list of discovered free models available via Kilo Gateway"""
    return {
        "object": "list",
        "data": FREE_MODELS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
