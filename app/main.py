from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from clerk_backend_api import Clerk, AuthenticateRequestOptions
from pydantic import BaseModel
import ollama
import os
from dotenv import load_dotenv
from typing import Optional
import httpx
from app.rag import RAG
# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="LLM API with Clerk Authentication")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Clerk
clerk = Clerk(bearer_auth=os.getenv('CLERK_SECRET_KEY'))

# Security scheme for JWT
security = HTTPBearer()
db_connection_string = os.getenv("DATABASE_URL")
rag = RAG(db_connection_string)

@app.on_event("startup")
async def startup_event():
    await rag.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await rag.close()

class LLMRequest(BaseModel):
    prompt: str
    model: str = "llama2"  # default model
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class LLMResponse(BaseModel):
    response: str
    model: str
    tokens_used: Optional[int]

async def verify_token(request: Request):
    try:
        # Convert FastAPI Request to httpx Request
        headers = dict(request.headers)
        method = request.method
        url = str(request.url)
        content = await request.body()
        
        httpx_request = httpx.Request(
            method=method,
            url=url,
            headers=headers,
            content=content
        )
        
        # Verify the request using Clerk's authenticate_request
        request_state = clerk.authenticate_request(
            httpx_request,
            AuthenticateRequestOptions(
                authorized_parties=['https://example.com']  # Replace with your domain
            )
        )
        
        if not request_state.is_signed_in:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return request_state
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: Request,
    llm_request: LLMRequest,
    # claims: dict = Depends(verify_token)
):
    try:
        # Use Ollama client to generate response
        response = await rag.generate_response(llm_request.prompt)
        
        return LLMResponse(
            response=response.get("response", ""),
            model=llm_request.model,
            tokens_used=response.get("eval_count")  # Ollama returns eval_count instead of tokens_used
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 