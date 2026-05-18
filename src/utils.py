import asyncio
import json
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

from config import (
    bilingual,
    camembert,
    claude,
    deepseek,
    gemini,
    gemini_pro,
    gpt,
    mistral,
    o3,
    o4,
    openrouter_key,
    opus,
)

_MODEL_CACHE: Dict[str, object] = {}

MODEL_ALIASES = {
    "o4": o4,
    "o3": o3,
    "gpt": gpt,
    "gemini_pro": gemini_pro,
    "gemini": gemini,
    "claude": claude,
    "opus": opus,
    "mistral": mistral,
    "deepseek": deepseek,
    "camembert": camembert,
    "bilingual": bilingual,
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Optional app-identification headers for OpenRouter leaderboard / analytics.
OPENROUTER_REFERER = None
OPENROUTER_TITLE = "joker"

# Default routing presets. Keep these conservative.
ROUTING_PRESETS = {
    "default": {
        "allow_fallbacks": True,
    },
    "fast": {
        "allow_fallbacks": True,
        # Example: uncomment if you want to pin preferred providers
        # "order": ["google-vertex", "fireworks", "together"]
    },
    "stable": {
        "allow_fallbacks": True,
    },
}

# Task-level defaults without changing call sites elsewhere.
TASK_PRESET_BY_MODEL_ALIAS = {
    "gemini": "fast",
    "gemini_pro": "stable",
    "gpt": "stable",
    "o3": "stable",
    "o4": "stable",
    "claude": "stable",
    "opus": "stable",
    "mistral": "fast",
    "deepseek": "fast",
}


def resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def get_api_model_str(model: str) -> str:
    return resolve_model_name(model)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def parse_json_response(raw_text: str) -> dict:
    raw_text = strip_code_fences(raw_text)

    # First try strict JSON parse.
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object in model response")
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback for models that wrap JSON with extra text.
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response")

    parsed = json.loads(raw_text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object in model response")
    return parsed


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        block_texts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    block_texts.append(text)
        if block_texts:
            return "\n".join(block_texts)

    return json.dumps(content, ensure_ascii=False, default=str)


def extract_text_from_openrouter_response(data: dict) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected OpenRouter response format: {data}") from e

    return normalize_message_content(message.get("content", ""))


def build_json_schema_response_format(schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": schema,
        },
    }


def build_object_schema_from_fields(fields: dict[str, str]) -> dict:
    """
    fields maps field name -> simple type: 'string', 'integer', 'number', 'boolean', 'array_string'
    """
    properties = {}
    required = []

    for name, field_type in fields.items():
        required.append(name)
        if field_type == "string":
            properties[name] = {"type": "string"}
        elif field_type == "integer":
            properties[name] = {"type": "integer"}
        elif field_type == "number":
            properties[name] = {"type": "number"}
        elif field_type == "boolean":
            properties[name] = {"type": "boolean"}
        elif field_type == "array_string":
            properties[name] = {
                "type": "array",
                "items": {"type": "string"},
            }
        else:
            raise ValueError(f"Unsupported field type: {field_type}")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def ensure_required_keys(payload: dict, required_keys: list[str]) -> dict:
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    return payload


class OpenRouterClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        referer: Optional[str] = OPENROUTER_REFERER,
        title: Optional[str] = OPENROUTER_TITLE,
    ):
        self.model = model
        self.api_key = api_key
        self.referer = referer
        self.title = title

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    def invoke(
        self,
        prompt: str,
        *,
        response_schema: Optional[dict] = None,
        routing_preset: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120,
        temperature: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
        }

        if response_schema is not None:
            payload["response_format"] = build_json_schema_response_format(response_schema)

        if temperature is not None:
            payload["temperature"] = temperature

        preset_name = routing_preset or "default"
        provider_settings = ROUTING_PRESETS.get(preset_name)
        if provider_settings:
            payload["provider"] = provider_settings

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OPENROUTER_URL,
                    headers=self._headers(),
                    json=payload,
                    timeout=timeout,
                )
                if response.status_code == 402:
                    try:
                        error_details = response.json()
                        print("--- RAW OPENROUTER 402 ERROR RESPONSE ---")
                        print(json.dumps(error_details, indent=2))
                        print("-----------------------------------------")
                    except json.JSONDecodeError:
                        print("--- RAW OPENROUTER 402 ERROR (NOT JSON) ---")
                        print(response.text)
                        print("-------------------------------------------")
                
                response.raise_for_status()
                data = response.json()
                return extract_text_from_openrouter_response(data)
            except requests.HTTPError as e:
                last_error = e
                status = getattr(e.response, "status_code", None)
                # Do not retry 402, let it fail fast as requested
                if status in {408, 409, 429, 500, 502, 503, 504} and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (requests.RequestException, ValueError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        raise last_error if last_error is not None else RuntimeError("Unknown OpenRouter error")

    async def ainvoke(
        self,
        prompt: str,
        *,
        response_schema: Optional[dict] = None,
        routing_preset: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120,
        temperature: Optional[float] = None,
    ) -> str:
        return await asyncio.to_thread(
            self.invoke,
            prompt,
            response_schema=response_schema,
            routing_preset=routing_preset,
            max_retries=max_retries,
            timeout=timeout,
            temperature=temperature,
        )


def get_model(model_str):
    model = get_api_model_str(model_str)
    cached_model = _MODEL_CACHE.get(model)
    if cached_model is not None:
        return cached_model

    if model == camembert:
        llm = SentenceTransformer(camembert)
    elif model == bilingual:
        llm = SentenceTransformer(bilingual, trust_remote_code=True)
    else:
        llm = OpenRouterClient(model=model, api_key=openrouter_key)

    _MODEL_CACHE[model] = llm
    return llm


def _routing_preset_for_model_alias(model_str: str) -> str:
    return TASK_PRESET_BY_MODEL_ALIAS.get(model_str, "default")


def get_response(
    prompt,
    model_str,
    *,
    response_schema: Optional[dict] = None,
    required_keys: Optional[list[str]] = None,
    routing_preset: Optional[str] = None,
    temperature: Optional[float] = None,
):
    llm = get_model(model_str)

    if isinstance(llm, SentenceTransformer):
        raise TypeError(
            f"Model '{model_str}' is an embedding model and cannot be used with get_response()"
        )

    response = llm.invoke(
        prompt,
        response_schema=response_schema,
        routing_preset=routing_preset or _routing_preset_for_model_alias(model_str),
        temperature=temperature,
    )
    response_json = parse_json_response(response)

    if required_keys:
        response_json = ensure_required_keys(response_json, required_keys)

    print(json.dumps(response_json, ensure_ascii=False))
    return pd.Series(response_json)


async def get_response_async(
    prompt,
    model_str,
    *,
    response_schema: Optional[dict] = None,
    required_keys: Optional[list[str]] = None,
    routing_preset: Optional[str] = None,
    temperature: Optional[float] = None,
):
    llm = get_model(model_str)

    if isinstance(llm, SentenceTransformer):
        raise TypeError(
            f"Model '{model_str}' is an embedding model and cannot be used with get_response_async()"
        )

    response = await llm.ainvoke(
        prompt,
        response_schema=response_schema,
        routing_preset=routing_preset or _routing_preset_for_model_alias(model_str),
        temperature=temperature,
    )
    response_json = parse_json_response(response)

    if required_keys:
        response_json = ensure_required_keys(response_json, required_keys)

    print(json.dumps(response_json, ensure_ascii=False))
    return pd.Series(response_json)


def get_response_not_json(
    prompt,
    model_str,
    *,
    routing_preset: Optional[str] = None,
    temperature: Optional[float] = None,
):
    llm = get_model(model_str)

    if isinstance(llm, SentenceTransformer):
        raise TypeError(
            f"Model '{model_str}' is an embedding model and cannot be used with get_response_not_json()"
        )

    return llm.invoke(
        prompt,
        routing_preset=routing_preset or _routing_preset_for_model_alias(model_str),
        temperature=temperature,
    )