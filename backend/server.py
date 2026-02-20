#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import spacy
from spacy.language import Language
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

DEFAULT_MODEL = "de_core_news_md"
DEFAULT_REDACT_CHAR = "█"

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


# Cache for loaded models
_models: dict[str, Language] = {}


def _get_model(model_name: str) -> Language:
    """Lazy-load a spaCy model, caching it for future use."""
    if model_name not in _models:
        try:
            _models[model_name] = spacy.load(
                model_name, disable=["lemmatizer", "attribute_ruler"]
            )
        except OSError:
            raise ValueError(
                f"Model '{model_name}' not found. Tell the admin to download it first."
            )
    return _models[model_name]


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


def redact(text: str, redaction_modes: list[ModeConfig], model_name: str, redact_char: str = "█") -> str:
    """Apply redaction to text based on specified modes and level."""
    threshold = {mode.type: (100 - mode.level) / 100.0 for mode in redaction_modes}
    nlp = _get_model(model_name)
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
            redact_segment(result, chunk.start_char, chunk.end_char, redact_char)

    if any(mode.type == RedactionMode.SINGLE for mode in redaction_modes):
        # redact individual tokens based on their importance and the specified level
        for token in doc:
            importance = get_token_importance(token)
            if importance < threshold[RedactionMode.SINGLE]:
                continue
            redact_segment(result, token.idx, token.idx + len(token), redact_char)

    return "".join(result)


def analyze(text: str, redaction_modes: list[ModeConfig], model_name: str) -> str:
    """Analyze importance of phrases and words based on specified modes and level."""
    threshold = {mode.type: (100 - mode.level) / 100.0 for mode in redaction_modes}
    nlp = _get_model(model_name)
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
            result.append([chunk.start_char, chunk.end_char, importance])

    if any(mode.type == RedactionMode.SINGLE for mode in redaction_modes):
        # redact individual tokens based on their importance and the specified level
        for token in doc:
            importance = get_token_importance(token)
            if importance < threshold[RedactionMode.SINGLE]:
                continue
            result.append([token.idx, token.idx + len(token), importance])

    if not result:
        return []

    # Sort by start index, then by importance descending.
    # This ensures that for overlapping segments, we process the one starting first.
    # If they start at the same index, the more important one comes first.
    result.sort(key=lambda x: (x[0], -x[2]))
    # Remove overlapping segments, keeping the one with higher importance.
    # If a less important segment is fully contained within a more important one, it's removed.
    merged = [result[0]]
    for current in result[1:]:
        last = merged[-1]
        # Check for overlap
        if current[0] < last[1]:
            # Overlap exists.
            # If current is fully contained in last, skip it.
            if current[1] <= last[1]:
                continue
            # Partial overlap. Decide which to keep.
            # If last is more important, adjust its end if current extends it.
            if last[2] >= current[2]:
                last[1] = max(last[1], current[1])
            else:
                # Current is more important, replace last.
                # This case is less likely with the current sorting, but good for robustness.
                merged[-1] = current
        else:
            # No overlap, add current segment.
            merged.append(current)
    result = merged
    # Final sort by start index
    result.sort(key=lambda x: x[0])
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models
    _models = {}
    yield
    # Cleanup
    _models = None


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
    return {"status": "healthy", "models": spacy.util.get_installed_models()}


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
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    try:
        analysis_result = analyze(request.text, request.modes, request.model)
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
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    model = request.model if hasattr(request, 'model') else DEFAULT_MODEL
    redact_char = request.redact_char if hasattr(request, 'redact_char') else DEFAULT_REDACT_CHAR
    try:
        redacted_text = redact(request.text, request.modes, model, redact_char)
        return JSONResponse(
            content={
                "redacted_text": redacted_text,
                "modes": [
                    {"type": mode.type, "level": mode.level} for mode in request.modes
                ],
                "model_used": model,
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
