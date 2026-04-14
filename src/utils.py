from config import bilingual, camembert, claude, deepseek, gemini, gemini_pro, gpt, mistral, o3, o4
from config import anthropic_key, deepseek_key, gemini_key, mistral_key, openai_key, openrouter_key
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

import asyncio
import pandas as pd
import json
from typing import Dict


_MODEL_CACHE: Dict[str, object] = {}

MODEL_ALIASES = {
  'o4': o4,
  'o3': o3,
  'gpt': gpt,
  'gemini_pro': gemini_pro,
  'gemini': gemini,
  'claude': claude,
  'mistral': mistral,
  'deepseek': deepseek,
  'camembert': camembert,
  'bilingual': bilingual,
}


def resolve_model_name(model):
  return MODEL_ALIASES.get(model, model)


def parse_json_response(raw_text):
  start = raw_text.find('{')
  end = raw_text.rfind('}')
  if start == -1 or end == -1 or end < start:
    raise ValueError('No JSON object found in model response')
  return json.loads(raw_text[start:end + 1])


def extract_text_from_ai_message(ai_msg):
  text_attr = getattr(ai_msg, 'text', None)
  text_value = None
  if callable(text_attr):
    try:
      text_value = text_attr()
    except TypeError:
      text_value = None
  else:
    text_value = text_attr

  if isinstance(text_value, str) and text_value:
    return text_value

  content = getattr(ai_msg, 'content', '')
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    block_texts = []
    for block in content:
      if isinstance(block, dict) and isinstance(block.get('text'), str):
        block_texts.append(block['text'])
    if block_texts:
      return "\n".join(block_texts)
  return json.dumps(content, default=str)

def get_response(prompt, model_str):
  llm = get_model(model_str)
  ai_msg = llm.invoke(prompt)
  response = extract_text_from_ai_message(ai_msg)
  response_json = parse_json_response(response)
  print(json.dumps(response_json, ensure_ascii=False))
  return pd.Series(response_json)


async def get_response_async(prompt, model_str):
  llm = get_model(model_str)
  if hasattr(llm, 'ainvoke'):
    ai_msg = await llm.ainvoke(prompt)
  else:
    ai_msg = await asyncio.to_thread(llm.invoke, prompt)
  response = extract_text_from_ai_message(ai_msg)
  response_json = parse_json_response(response)
  print(json.dumps(response_json, ensure_ascii=False))
  return pd.Series(response_json)


def get_response_not_json(prompt, model):
  llm = get_model(model)
  ai_msg = llm.invoke(prompt)
  return extract_text_from_ai_message(ai_msg)


def get_model(model_str):
  model = get_api_model_str(model_str)
  cached_model = _MODEL_CACHE.get(model)
  if cached_model is not None:
    return cached_model

  if model == camembert:
    from sentence_transformers import SentenceTransformer
    llm = SentenceTransformer(camembert)
  elif model == bilingual:
    from sentence_transformers import SentenceTransformer
    llm = SentenceTransformer(bilingual, trust_remote_code=True)
  elif model == o4 or model == o3 or model == gpt:
    llm = ChatOpenAI(model=model, api_key=openai_key)
  elif model == claude or model.startswith('claude'):
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=model, api_key=anthropic_key)
  elif model == mistral or model.startswith('mistral'):
    from langchain_mistralai import ChatMistralAI
    llm = ChatMistralAI(model=model, api_key=mistral_key)
  elif model == deepseek or model.startswith('deepseek'):
    from langchain_deepseek import ChatDeepSeek
    llm = ChatDeepSeek(model=model, api_key=deepseek_key)
  elif model == gemini_pro or model == gemini or model.startswith('gemini'):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm_kwargs = {'model': model, 'api_key': gemini_key, 'max_retries': 2}
    if model.startswith('gemini-3'):
      llm_kwargs['temperature'] = 1.0
    llm = ChatGoogleGenerativeAI(**llm_kwargs)
  else:
    llm = ChatOpenRouter(model=model, api_key=openrouter_key)

  _MODEL_CACHE[model] = llm
  return llm


def get_api_model_str(model):
  return resolve_model_name(model)


class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self, model: str, api_key: str, openai_api_base: str = "https://openrouter.ai/api/v1"):
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=api_key,
            model_name=model,
        )
