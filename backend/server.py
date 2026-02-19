#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import spacy
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field


class RedactionMode(str, Enum):
    PHRASE = "phrase"
    SINGLE = "single"


class ModeConfig(BaseModel):
    type: RedactionMode = Field(..., description="Redaction mode type")
    level: int = Field(80, ge=0, le=100, description="Redaction level (0-100)")


class RedactionRequest(BaseModel):
    text: str = Field(..., description="Text to redact")
    modes: list[ModeConfig] = Field(
        [ModeConfig(type=RedactionMode.PHRASE, level=80)],
        description="Redaction mode(s)",
    )
    model: str = Field("de_core_news_md", description="spaCy model to use")


class RedactionResponse(BaseModel):
    original_text: str
    redacted_text: str
    modes: list[ModeConfig]
    model_used: str


# Global spaCy model
nlp = None


def redact_segment(
    result: list, start_idx: int, end_idx: int, redact_char: str = "█"
) -> None:
    """Redact a segment of text by replacing non-whitespace characters."""
    result[start_idx:end_idx] = [
        redact_char if not c.isspace() else c for c in result[start_idx:end_idx]
    ]


def get_token_importance(token) -> float:
    """Calculate importance score for a token."""
    if token.ent_type_:
        return 1.0
    elif token.pos_ == "NUM":
        return 0.9
    elif token.pos_ in ["NOUN", "PROPN"]:
        return 0.7
    elif token.pos_ in ["VERB", "ADJ", "ADV"]:
        return 0.4
    else:
        return 0.0


def redact(text: str, redaction_modes: list[ModeConfig]) -> str:
    """Apply redaction to text based on specified modes and level."""
    threshold = {mode.type: (100 - mode.level) / 100.0 for mode in redaction_modes}
    doc = nlp(text)
    result = list(text)

    if any(mode.type == RedactionMode.PHRASE for mode in redaction_modes):
        # redact entire noun phrases based on the specified level
        for chunk in doc.noun_chunks:
            tokens = [tok for tok in chunk if not tok.is_stop]
            if not tokens:
                continue
            importance = sum(get_token_importance(tok) for tok in tokens) / len(tokens)
            if importance < threshold[RedactionMode.PHRASE]:
                continue
            redact_segment(result, chunk.start_char, chunk.end_char)

    if any(mode.type == RedactionMode.SINGLE for mode in redaction_modes):
        # redact individual tokens based on their importance and the specified level
        for token in doc:
            importance = get_token_importance(token)
            if importance < threshold[RedactionMode.SINGLE]:
                continue
            redact_segment(result, token.idx, token.idx + len(token), "▒")

    return "".join(result)

def analyze(text: str, redaction_modes: list[ModeConfig]) -> str:
    """Analyze importance of phrases and words based on specified modes and level."""
    threshold = {mode.type: (100 - mode.level) / 100.0 for mode in redaction_modes}
    doc = nlp(text)

    result = []

    if any(mode.type == RedactionMode.PHRASE for mode in redaction_modes):
        # redact entire noun phrases based on the specified level
        for chunk in doc.noun_chunks:
            tokens = [tok for tok in chunk if not tok.is_stop]
            if not tokens:
                continue
            importance = sum(get_token_importance(tok) for tok in tokens) / len(tokens)
            if importance < threshold[RedactionMode.PHRASE]:
                continue
            result.append([chunk.start_char, chunk.end_char, importance, 'P'])

    if any(mode.type == RedactionMode.SINGLE for mode in redaction_modes):
        # redact individual tokens based on their importance and the specified level
        for token in doc:
            importance = get_token_importance(token)
            if importance < threshold[RedactionMode.SINGLE]:
                continue
            result.append([token.idx, token.idx + len(token), importance, 'S'])

    if result:
        # Remove overlapping segments, keeping the one with higher importance
        # Since we sorted by importance descending for the same start index,
        # the first one we encounter is the one to keep.
        merged = [result[0]]
        for current in result[1:]:
            last = merged[-1]
            # Check for overlap: current segment starts before the last one ends
            if current[0] < last[1]:
                # If the current segment is more important and doesn't just start within the last one,
                # it might be a candidate to replace or merge, but our primary sort handles this.
                # We simply skip overlapping segments that start later but are less or equally important.
                # If an inner segment is more important, it should have been sorted first.
                # This logic keeps the first-encountered (most important or longest) segment in an overlapping group.
                pass
            else:
                # No overlap, add the current segment
                merged.append(current)
        
        result = merged
        # Final sort by start index
        result.sort(key=lambda x: x[0])

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load spaCy model on server startup."""
    global nlp
    print("Loading spaCy model...")
    nlp = spacy.load("de_core_news_md", disable=["lemmatizer", "attribute_ruler"])
    print("Model loaded successfully!")
    yield
    # Cleanup
    nlp = None


# Initialize FastAPI app
app = FastAPI(
    title="Text Redaction API",
    description="A KI-powered text redaction service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Text Redaction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Analyze text importance without redaction",
            "POST /redact": "Redact text with specified parameters",
            "GET /health": "Check API health status",
            "GET /models": "List available spaCy models",
        },
    }

@app.options("/health")
async def health_options():
    """Handle CORS preflight requests."""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.options("/models")
async def models_options():
    """Handle CORS preflight requests."""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.get("/models")
async def get_models():
    return {"models": ["de_core_news_md"]}

@app.options("/redact")
async def redact_options():
    """Handle CORS preflight requests."""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.post("/analyze")
async def analyze_text(request: RedactionRequest):
    """
    Analyze importance of phrases and words in the provided text.

    - **text**: The text to analyze
    - **level**: Analysis aggressiveness (0-100, higher = more analysis)
    - **mode**: Analysis strategy (phrase, single, or both)
    - **model**: spaCy model to use (default: de_core_news_md)
    """
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    try:
        analysis_result = analyze(request.text, request.modes)
        return JSONResponse(
            content={
                "analysis": analysis_result,
                "modes": [
                    {"type": mode.type, "level": mode.level} for mode in request.modes
                ],
                "model_used": request.model,
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/redact")
async def redact_text(request: RedactionRequest):
    """
    Redact sensitive information from the provided text.

    - **text**: The text to redact
    - **level**: Redaction aggressiveness (0-100, higher = more redaction)
    - **mode**: Redaction strategy (phrase, single, or both)
    - **model**: spaCy model to use (default: de_core_news_md)
    """
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    try:
        redacted_text = redact(request.text, request.modes)
        return JSONResponse(
            content={
                "redacted_text": redacted_text,
                "modes": [
                    {"type": mode.type, "level": mode.level} for mode in request.modes
                ],
                "model_used": request.model,
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )
    except Exception as e:
        print(f"Error during redaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Redaction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=13370)
