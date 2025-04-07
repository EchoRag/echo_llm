# LLM API with Clerk Authentication

This is a FastAPI application that provides a secure API endpoint for interacting with Ollama's LLM models. The API is protected using Clerk for JWT authentication.

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Clerk account and API credentials

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your credentials:
   ```
   CLERK_SECRET_KEY=your_clerk_secret_key
   OLLAMA_BASE_URL=http://localhost:11434
   ```
4. Make sure Ollama is running locally with the desired model (e.g., llama2)

## Running the Application

Start the server with:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /generate
Generate text using the specified LLM model.

Request body:
```json
{
    "prompt": "Your prompt here",
    "model": "llama2",
    "max_tokens": 1000,
    "temperature": 0.7
}
```

Headers:
```
Authorization: Bearer your_jwt_token
```

### GET /health
Health check endpoint.

## Authentication

The API uses Clerk for JWT authentication. You need to:
1. Sign up for a Clerk account
2. Create a new application in Clerk
3. Get your secret key from Clerk dashboard
4. Add the secret key to your `.env` file

## Error Handling

The API includes proper error handling for:
- Invalid authentication
- Ollama API errors
- General server errors

## Security

- All endpoints except `/health` require JWT authentication
- Tokens are verified using Clerk's backend API
- Environment variables are used for sensitive data 