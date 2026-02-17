# FastAPI Text Redaction Server

## Installation

```bash
pipenv install
```

## Running the Server

```bash
pipenv run python server.py
```

The server will start on `http://0.0.0.0:8000`

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### GET /
Returns API information and available endpoints.

**Response:**
```json
{
  "service": "Text Redaction API",
  "version": "1.0.0",
  "endpoints": {
    "POST /redact": "Redact text with specified parameters",
    "GET /health": "Check API health status"
  }
}
```

### GET /health
Health check endpoint to verify the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /redact
Redact sensitive information from provided text.

**Request Body:**
```json
{
  "text": "Your text to redact",
  "level": 80,
  "mode": ["phrase", "single"],
  "model": "de_core_news_md"
}
```

**Parameters:**
- `text` (string, required): The text to redact
- `level` (integer, 0-100): Redaction aggressiveness. Higher values mean more aggressive redaction (default: 80)
- `mode` (array): Redaction strategy. Options: "phrase" (entire noun phrases), "single" (individual tokens), or both (default: ["phrase"])
- `model` (string): spaCy model to use (default: "de_core_news_md")

**Response:**
```json
{
  "original_text": "Your text to redact",
  "redacted_text": "███ ████ ██ ██████",
  "level": 80,
  "modes": ["phrase", "single"],
  "model_used": "de_core_news_md"
}
```

## Example Usage

### Using curl

```bash
curl -X POST http://localhost:8000/redact \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Smith lives in Berlin and works at Google",
    "level": 80,
    "mode": ["phrase", "single"]
  }'
```

### Using Python requests

```python
import requests

response = requests.post(
    "http://localhost:8000/redact",
    json={
        "text": "John Smith lives in Berlin and works at Google",
        "level": 80,
        "mode": ["phrase", "single"]
    }
)
print(response.json())
```

### Using JavaScript fetch

```javascript
const response = await fetch('http://localhost:8000/redact', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'John Smith lives in Berlin and works at Google',
    level: 80,
    mode: ['phrase', 'single']
  })
});
const data = await response.json();
console.log(data);
```

## Key Changes from CLI Version

1. **Pydantic Models**: Request/response data is validated with Pydantic models
2. **Async/Await**: Endpoints are async for better performance
3. **Error Handling**: HTTP exceptions with appropriate status codes
4. **Auto Documentation**: Swagger UI and ReDoc available at `/docs` and `/redoc`
5. **Health Checks**: `/health` endpoint for monitoring
6. **Global Model**: spaCy model is loaded once at startup
7. **Flexible API**: Can specify redaction mode and level per request
