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

trace.set_tracer_provider(TracerProvider(resource=resource))
otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    insecure=True
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

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
    user_message_id: Optional[str]
    assistant_message_id: Optional[str]

class ConversationList(BaseModel):
    conversations: List[Dict[str, Any]]

class VoteRequest(BaseModel):
    # message_id: str
    vote_type: str  # 'like' or 'dislike'

class VoteResponse(BaseModel):
    upvotes: int
    downvotes: int

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
                llm_request.prompt,
                llm_request.conversation_id
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

@app.post("/messages/{message_id}/vote", response_model=VoteResponse)
async def vote_message(
    message_id: str,
    vote_request: VoteRequest,
    auth: dict = Depends(verify_token)
):
    try:
        if vote_request.vote_type not in ['like', 'dislike']:
            raise HTTPException(status_code=400, detail="Invalid vote type. Must be 'like' or 'dislike'")
            
        # Convert like/dislike to upvote/downvote for database
        db_vote_type = 'upvote' if vote_request.vote_type == 'like' else 'downvote'
            
        result = await rag.db.vote_message(
            message_id=message_id,
            user_provider_uid=auth["user_id"],
            vote_type=db_vote_type
        )
        
        return VoteResponse(**result)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing vote: {str(e)}"
        ) 