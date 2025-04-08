from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from clerk_backend_api import Clerk
from pydantic import BaseModel
from jose import jwt, jwk
from jose.exceptions import JWTError
import ollama
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import requests
import logging
from app.rag import RAG

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="LLM API with Clerk Authentication")

# Clerk configuration
CLERK_SECRET_KEY = os.getenv('CLERK_SECRET_KEY')
CLERK_JWKS_URL = os.getenv('CLERK_JWKS_URL')
CLERK_ISSUER = os.getenv('CLERK_ISSUER')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Clerk
clerk = Clerk(bearer_auth=CLERK_SECRET_KEY)

# Security scheme for JWT
security = HTTPBearer()

def get_jwks():
    response = requests.get(CLERK_JWKS_URL)
    return response.json()

def get_public_key(kid):
    jwks = get_jwks()
    for key in jwks['keys']:
        if key['kid'] == kid:
            return jwk.construct(key)
    raise HTTPException(status_code=401, detail="Invalid token")

def decode_token(token: str):
    try:
        headers = jwt.get_unverified_headers(token)
        kid = headers['kid']
        public_key = get_public_key(kid)
        return jwt.decode(
            token, 
            public_key.to_pem().decode('utf-8'), 
            algorithms=['RS256'], 
            issuer=CLERK_ISSUER
        )
    except JWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = decode_token(token)
        user_id = payload.get('sub')
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
            
        
        return {
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Database setup
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
    conversation_id: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    model: str
    tokens_used: Optional[int]
    conversation_id: Optional[str]

class ConversationList(BaseModel):
    conversations: List[Dict[str, Any]]

@app.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: Request,
    llm_request: LLMRequest,
    auth: dict = Depends(verify_token)
):
    try:
        # Get user from auth
        user_provider_uid = auth["user_id"]
        
        # Create new conversation if no conversation_id provided
        if not llm_request.conversation_id:
            llm_request.conversation_id = await rag.db.create_conversation(user_provider_uid)
        
        # Use RAG to generate response
        response = await rag.generate_response(
            llm_request.prompt,
            llm_request.conversation_id
        )
        
        return LLMResponse(
            response=response.get("response", ""),
            model=llm_request.model,
            tokens_used=response.get("eval_count"),
            conversation_id=llm_request.conversation_id
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/conversations", response_model=ConversationList)
async def get_conversations(
    request: Request,
    auth: dict = Depends(verify_token)
):
    try:
        # Get user from auth
        user_provider_uid = auth["user_id"]
        
        # Get user's conversations
        conversations = await rag.db.get_user_conversations(user_provider_uid)
        
        return ConversationList(conversations=conversations)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching conversations: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 