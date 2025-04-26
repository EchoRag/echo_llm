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

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenTelemetry
resource = Resource(attributes={
    "service.name": "llm-api",
    "service.version": "1.0.0"
})

# Get service token from environment
SERVICE_TOKEN = os.getenv('SERVICE_TOKEN')
if not SERVICE_TOKEN:
    raise ValueError("SERVICE_TOKEN environment variable is required")

# Initialize FastAPI app
app = FastAPI(title="LLM API with Clerk Authentication")

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

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

# Get tracer
tracer = trace.get_tracer(__name__)

def get_jwks():
    with tracer.start_as_current_span("get_jwks"):
        response = requests.get(CLERK_JWKS_URL)
        return response.json()

def get_public_key(kid):
    with tracer.start_as_current_span("get_public_key"):
        jwks = get_jwks()
        for key in jwks['keys']:
            if key['kid'] == kid:
                return jwk.construct(key)
        raise HTTPException(status_code=401, detail="Invalid token")

def decode_token(token: str):
    with tracer.start_as_current_span("decode_token"):
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
    with tracer.start_as_current_span("verify_token"):
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

async def register_with_proxy_server():
    """
    Register this LLM server with the proxy server
    """
    try:
        # Get the server URL from environment or use default
        server_url = 'http://localhost:8001'
        # Get the public IP address
        try:
            response = requests.get('https://api.ipify.org?format=json')
            if response.status_code == 200:
                public_ip = response.json()['ip']
                server_url = f"http://{public_ip}:8001"  # Using port 8001 as specified in server.py
                logger.info(f"Using public IP address: {public_ip}")
            else:
                logger.warning("Failed to get public IP, using default server URL")
        except Exception as e:
            logger.error(f"Error getting public IP: {str(e)}")
            logger.warning("Using default server URL")
        # Make the registration request
        response = requests.post(
            f"{os.getenv('API_URL')}/api/v1/proxy-server/register",
            json={'llmServerUrl': server_url},
            headers={'Authorization': f'Bearer {SERVICE_TOKEN}'}
        )
        
        if response.status_code == 200:
            logger.info("Successfully registered with proxy server")
            return response.json()
        else:
            logger.error(f"Failed to register with proxy server: {response.text}")
            raise Exception(f"Proxy server registration failed: {response.text}")
            
    except Exception as e:
        logger.error(f"Error registering with proxy server: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    await rag.initialize()
    await register_with_proxy_server()

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
    user_message_id: Optional[str]
    assistant_message_id: Optional[str]

class ConversationList(BaseModel):
    conversations: List[Dict[str, Any]]

@app.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: Request,
    llm_request: LLMRequest,
    auth: dict = Depends(verify_token)
):
    with tracer.start_as_current_span("generate_text"):
        try:
            user_provider_uid = auth["user_id"]
            
            if not llm_request.conversation_id:
                llm_request.conversation_id = await rag.db.create_conversation(user_provider_uid)
            
            response = await rag.generate_response(
                prompt=llm_request.prompt,
                user_provider_uid=user_provider_uid,
                conversation_id=llm_request.conversation_id
            )
            
            return LLMResponse(
                response=response.get("response", ""),
                model=llm_request.model,
                tokens_used=response.get("eval_count"),
                conversation_id=llm_request.conversation_id,
                user_message_id=response.get("user_message_id"),
                assistant_message_id=response.get("assistant_message_id")
            )
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

@app.get("/health")
async def health_check():
    with tracer.start_as_current_span("health_check"):
        return {"status": "healthy"}

