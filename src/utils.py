import asyncio
import json
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

from config import (
    MODEL_ALIASES,
    bilingual,
    camembert,
    openrouter_key,
)

_MODEL_CACHE: Dict[str, object] = {}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = None
OPENROUTER_TITLE = "joker"

ROUTING_PRESETS = {
    "default": {"allow_fallbacks": True},
    "fast": {"allow_fallbacks": True},
    "stable": {"allow_fallbacks": True},
}

TASK_PRESET_BY_MODEL_ALIAS = {
    "gemini": "fast",
    "gemini_pro": "stable",
    "gpt": "stable",
    "claude": "stable",
    "deepseek": "fast",
    "qwen": "fast",
    "o3": "stable",
    "o4": "stable",
    "opus": "stable",
    "mistral": "fast",
}

ENABLE_RESPONSE_HEALING = True


def resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def get_api_model_str(model: str) -> str:
    return resolve_model_name(model)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            first = lines[0].strip().lower()
            if first in {"```", "```json", "```javascript", "```js"}:
                return "\n".join(lines[1:-1]).strip()
    return text


def parse_json_response(raw_text: str) -> dict:
    raw_text = strip_code_fences(raw_text)

    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            preview = raw_text[:1000].replace("\n", "\\n")
            raise ValueError(
                f"Expected JSON object in model response, got {type(parsed).__name__}. "
                f"Raw preview: {preview}"
            )
        return parsed
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        preview = raw_text[:1000].replace("\n", "\\n")
        raise ValueError(f"No JSON object found in model response. Raw preview: {preview}")

    try:
        parsed = json.loads(raw_text[start:end + 1])
    except json.JSONDecodeError as e:
        preview = raw_text[:1000].replace("\n", "\\n")
        raise ValueError(f"Invalid JSON object in model response: {e}. Raw preview: {preview}") from e

    if not isinstance(parsed, dict):
        preview = raw_text[:1000].replace("\n", "\\n")
        raise ValueError(
            f"Expected JSON object in model response, got {type(parsed).__name__}. "
            f"Raw preview: {preview}"
        )
    return parsed


def normalize_message_content(content: Any) -> str:
    """Normalize OpenRouter/OpenAI/Anthropic content into a JSON string.

    Some providers return structured content blocks instead of a plain string,
    e.g. [{"type": "json", "json": {...}}] or [{"parsed": {...}}].
    The old code serialized the whole list, producing a top-level JSON array and
    then failing with "Expected JSON object in model response". This function
    extracts the actual JSON object when present.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        for key in ("json", "parsed", "object", "data"):
            value = content.get(key)
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, str):
                return value

        if "candidates" in content:
            return json.dumps(content, ensure_ascii=False, separators=(",", ":"))

        text = content.get("text")
        if isinstance(text, str):
            return text

        return json.dumps(content, ensure_ascii=False, default=str)

    if isinstance(content, list):
        # Prefer an explicit JSON/object block.
        for block in content:
            if isinstance(block, dict):
                for key in ("json", "parsed", "object", "data"):
                    value = block.get(key)
                    if isinstance(value, dict):
                        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, str):
                        return value

                if "candidates" in block:
                    return json.dumps(block, ensure_ascii=False, separators=(",", ":"))

        # Then accept text blocks, if present.
        block_texts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    block_texts.append(text)

                content_value = block.get("content")
                if isinstance(content_value, str):
                    block_texts.append(content_value)
                elif isinstance(content_value, dict):
                    return json.dumps(content_value, ensure_ascii=False, separators=(",", ":"))

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
            "name": "joker_generation_response",
            "strict": True,
            "schema": schema,
        },
    }


def build_json_object_response_format() -> dict:
    return {"type": "json_object"}


def build_object_schema_from_fields(fields: dict[str, str]) -> dict:
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
            properties[name] = {"type": "array", "items": {"type": "string"}}
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
        timeout: int = 180,
        temperature: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You must return exactly one valid JSON object. "
                        "Do not include markdown, code fences, commentary, or text outside JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 5000,
        }

        if response_schema is not None:
            payload["response_format"] = build_json_schema_response_format(response_schema)
            if ENABLE_RESPONSE_HEALING:
                payload["plugins"] = [{"id": "response-healing"}]
        else:
            payload["response_format"] = build_json_object_response_format()

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

                if response.status_code >= 400:
                    try:
                        error_details = response.json()
                        error_text = json.dumps(error_details, ensure_ascii=False)
                    except Exception:
                        error_text = response.text
                    print("--- RAW OPENROUTER ERROR RESPONSE ---")
                    print(error_text[:4000])
                    print("-----------------------------------")

                response.raise_for_status()
                data = response.json()
                return extract_text_from_openrouter_response(data)

            except requests.HTTPError as e:
                last_error = e
                status = getattr(e.response, "status_code", None)
                if status == 400:
                    raise
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
        timeout: int = 180,
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
        raise TypeError(f"Model '{model_str}' is an embedding model and cannot be used with get_response()")

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
        raise TypeError(f"Model '{model_str}' is an embedding model and cannot be used with get_response_async()")

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
    raise RuntimeError(
        "Non-JSON OpenRouter responses are disabled for this pipeline. "
        "Use get_response()/get_response_async() with response_format JSON enforcement."
    )
