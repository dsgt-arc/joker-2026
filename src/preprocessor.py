import asyncio
import ast
import os
import sys
from typing import Any, Awaitable, Callable

import pandas as pd

from data import load, load_all, save
from config import combined_en_path, homonym_dir, identify_dir, translate_dir, similarity_dir
from utils import get_model, get_response_async

pd.options.mode.chained_assignment = None

DEFAULT_MODEL = os.environ.get("PREPROCESSOR_DEFAULT_MODEL", "google/gemini-3-pro")
VERBOSE = os.environ.get("PREPROCESSOR_VERBOSE", "1") == "1"
MAX_CONCURRENCY = int(os.environ.get("PREPROCESSOR_MAX_CONCURRENCY", "8"))


def log(*args):
    if VERBOSE:
        print(*args)


def safe_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            value = ast.literal_eval(x)
            return value if isinstance(value, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def log_and_build_fallback(error: Exception, payload: dict[str, Any]) -> pd.Series:
    print(f"Error: {error}")
    return pd.Series(payload)


async def run_async_apply(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[Any]],
    result_columns: list[str],
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(index, row):
        async with semaphore:
            result = await apply_async_fn(row)
            return index, result

    tasks = [asyncio.create_task(worker(index, row)) for index, row in chunk_df.iterrows()]
    results = {}

    try:
        for task in asyncio.as_completed(tasks):
            index, result = await task
            results[index] = result
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    ordered_rows = [results[index] for index in chunk_df.index]
    result_df = pd.DataFrame(ordered_rows, index=chunk_df.index)
    return result_df[result_columns]


async def run_async_chunk(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[Any]],
    result_columns: list[str],
) -> pd.DataFrame:
    return await run_async_apply(chunk_df, apply_async_fn, result_columns)


async def identify_pun_meanings(df, model, start=0, end=-1):
    output_columns = [
        "pun_word",
        "pun_type",
        "first_meaning",
        "second_meaning",
    ]

    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pun_word": {"type": "string"},
            "pun_type": {"type": "string"},
            "first_meaning": {"type": "array", "items": {"type": "string"}},
            "second_meaning": {"type": "array", "items": {"type": "string"}},
        },
        "required": output_columns,
    }

    async def apply(row):
        text_clean = row["text_clean"]

        prompt = f"""
Text: {text_clean}

Step 1: Identify the pun word in this text. Output one word.
Step 2: Determine whether the pun is homographic or homophonic. Output either "homographic" or "homophonic".
Step 3: Give a list of synonyms for each of the two meanings of the pun. If it is a homophonic pun, include the relevant homophones in the appropriate lists.

Return only valid JSON.
"""

        log(row.name, text_clean)
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=response_schema,
                required_keys=output_columns,
                routing_preset="stable",
            )
        except Exception as e:
            response = log_and_build_fallback(
                e,
                {
                    "pun_word": "ERROR",
                    "pun_type": "",
                    "first_meaning": [],
                    "second_meaning": [],
                },
            )
        return response

    chunk_size = 100
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)

    for i in range(start, end):
        chunks[i][output_columns] = await run_async_chunk(chunks[i], apply, output_columns)
        save(chunks[i], f"{identify_dir}{model}/{i}.tsv")


async def translate_pun_meanings(df, model, start=0, end=-1, translate_flag=True):
    fr_columns = [
        "pun_word_fr",
        "first_meaning_fr",
        "second_meaning_fr",
    ]
    bt_columns = [
        "pun_word_bt",
        "first_meaning_bt",
        "second_meaning_bt",
    ]

    fr_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pun_word_fr": {"type": "string"},
            "first_meaning_fr": {"type": "array", "items": {"type": "string"}},
            "second_meaning_fr": {"type": "array", "items": {"type": "string"}},
        },
        "required": fr_columns,
    }

    bt_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pun_word_bt": {"type": "string"},
            "first_meaning_bt": {"type": "array", "items": {"type": "string"}},
            "second_meaning_bt": {"type": "array", "items": {"type": "string"}},
        },
        "required": bt_columns,
    }

    async def translate(row):
        row_dict = row.to_dict()
        payload = {
            "pun_word_fr": row_dict["pun_word"],
            "first_meaning_fr": safe_list(row_dict["first_meaning"]),
            "second_meaning_fr": safe_list(row_dict["second_meaning"]),
        }

        prompt = f"""
Translate only the VALUES of this JSON object from English to French.
Do not change the keys.
Preserve the structure exactly.
If a value is a list, translate each element.

Input JSON:
{payload}

Return only valid JSON.
"""

        log(
            row.name,
            payload["pun_word_fr"],
            payload["first_meaning_fr"],
            payload["second_meaning_fr"],
        )
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=fr_schema,
                required_keys=fr_columns,
                routing_preset="fast",
            )
        except Exception as e:
            response = log_and_build_fallback(
                e,
                {
                    "pun_word_fr": "ERROR",
                    "first_meaning_fr": [],
                    "second_meaning_fr": [],
                },
            )
        return response

    async def back_translate(row):
        payload = {
            "pun_word_bt": row["pun_word_fr"],
            "first_meaning_bt": safe_list(row["first_meaning_fr"]),
            "second_meaning_bt": safe_list(row["second_meaning_fr"]),
        }

        prompt = f"""
Translate only the VALUES of this JSON object from French to English.
Do not change the keys.
Preserve the structure exactly.
If a value is a list, translate each element.

Input JSON:
{payload}

Return only valid JSON.
"""

        log(
            row.name,
            payload["pun_word_bt"],
            payload["first_meaning_bt"],
            payload["second_meaning_bt"],
        )
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=bt_schema,
                required_keys=bt_columns,
                routing_preset="fast",
            )
        except Exception as e:
            response = log_and_build_fallback(
                e,
                {
                    "pun_word_bt": "ERROR",
                    "first_meaning_bt": [],
                    "second_meaning_bt": [],
                },
            )
        return response

    chunk_size = 100
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    end = end if end > 0 else len(chunks)

    for i in range(start, end):
        if translate_flag:
            chunks[i][fr_columns] = await run_async_chunk(chunks[i], translate, fr_columns)
            save(chunks[i], f"{translate_dir}{model}/t/{i}.tsv")

        translate_df = load(f"{translate_dir}{model}/t/{i}.tsv")
        translate_df[bt_columns] = await run_async_chunk(translate_df, back_translate, bt_columns)
        save(translate_df, f"{translate_dir}{model}/{i}.tsv")


def get_cosine_similarity(df, model, start=0, end=-1):
    import torch
    from sentence_transformers import util

    def mean_embedding_or_zero(st_model, values):
        values = safe_list(values)
        if not values:
            dim = st_model.get_sentence_embedding_dimension()
            return torch.zeros((1, dim))
        return torch.mean(st_model.encode(values, convert_to_tensor=True), dim=0, keepdim=True)

    def apply(row, st_model):
        pun_word_embedding_en = st_model.encode([row["pun_word"]], convert_to_tensor=True)
        first_meaning_embedding_en = mean_embedding_or_zero(st_model, row["first_meaning"])
        second_meaning_embedding_en = mean_embedding_or_zero(st_model, row["second_meaning"])

        pun_word_embedding_fr = st_model.encode([row["pun_word_fr"]], convert_to_tensor=True)
        first_meaning_embedding_fr = mean_embedding_or_zero(st_model, row["first_meaning_fr"])
        second_meaning_embedding_fr = mean_embedding_or_zero(st_model, row["second_meaning_fr"])

        first_similarity_en = util.cos_sim(pun_word_embedding_en, first_meaning_embedding_en).item()
        second_similarity_en = util.cos_sim(pun_word_embedding_en, second_meaning_embedding_en).item()
        first_similarity_fr = util.cos_sim(pun_word_embedding_fr, first_meaning_embedding_fr).item()
        second_similarity_fr = util.cos_sim(pun_word_embedding_fr, second_meaning_embedding_fr).item()

        first_similarity_diff = first_similarity_en - first_similarity_fr
        second_similarity_diff = second_similarity_en - second_similarity_fr

        log(row.name, row["pun_word"], row["pun_word_fr"], row["pun_type"])
        log("first en", first_similarity_en, "fr", first_similarity_fr, "diff", first_similarity_diff)
        log("second en", second_similarity_en, "fr", second_similarity_fr, "diff", second_similarity_diff)

        result = {
            "first_similarity_en": first_similarity_en,
            "second_similarity_en": second_similarity_en,
            "first_similarity_fr": first_similarity_fr,
            "second_similarity_fr": second_similarity_fr,
            "first_similarity_diff": first_similarity_diff,
            "second_similarity_diff": second_similarity_diff,
        }
        return pd.Series(result)

    st_model = get_model(model)
    chunk_size = 10
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)

    for i in range(start, end):
        current_df = chunks[i]
        current_df[
            [
                "first_similarity_en",
                "second_similarity_en",
                "first_similarity_fr",
                "second_similarity_fr",
                "first_similarity_diff",
                "second_similarity_diff",
            ]
        ] = current_df.apply(apply, axis=1, args=(st_model,))
        save(current_df, f"{similarity_dir}{model}/{i}.tsv")


async def check_french_homonyms(df, model, start=0, end=-1):
    output_columns = ["is_homonym", "first_meaning_overlap", "second_meaning_overlap"]

    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_homonym": {"type": "integer"},
            "first_meaning_overlap": {"type": "integer"},
            "second_meaning_overlap": {"type": "integer"},
        },
        "required": output_columns,
    }

    async def apply(row):
        pun_word_fr = row["pun_word_fr"]
        first_meaning_fr = safe_list(row["first_meaning_fr"])
        second_meaning_fr = safe_list(row["second_meaning_fr"])

        prompt = f"""
Question 1: Is the French word "{pun_word_fr}" a homonym? Output 1 for yes or 0 for no.
Question 2: Does the word "{pun_word_fr}" share at least one meaning with any word in this list: {first_meaning_fr}? Output 1 for yes or 0 for no.
Question 3: Does the word "{pun_word_fr}" share at least one meaning with any word in this list: {second_meaning_fr}? Output 1 for yes or 0 for no.

Return only valid JSON.
"""

        log(row.name, pun_word_fr, first_meaning_fr, second_meaning_fr)
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=response_schema,
                required_keys=output_columns,
                routing_preset="fast",
            )
        except Exception as e:
            response = log_and_build_fallback(
                e,
                {
                    "is_homonym": -1,
                    "first_meaning_overlap": -1,
                    "second_meaning_overlap": -1,
                },
            )
        return response

    chunk_size = 10
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)

    for i in range(start, end):
        chunks[i][output_columns] = await run_async_chunk(chunks[i], apply, output_columns)
        save(chunks[i], f"{homonym_dir}{model}/{i}.tsv")


def generate_french_puns(df):
    return True


async def main():
    task = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    translate_flag = False if len(sys.argv) > 5 else True

    if task == "identify":
        df = load(combined_en_path)
        await identify_pun_meanings(df, model, start, end)

    if task == "translate":
        df = load_all(f"{identify_dir}gemini/")
        save(df, f"{identify_dir}gemini.tsv")
        await translate_pun_meanings(df, model, start, end, translate_flag)

    # if task == "similarity":
    #     df = load_all(f"{translate_dir}o4/t/")
    #     save(df, f"{translate_dir}o4.tsv")
    #     get_cosine_similarity(df, model, start, end)

    # if task == "homonym":
    #     df = load_all(f"{similarity_dir}bilingual/")
    #     save(df, f"{similarity_dir}bilingual.tsv")
    #     await check_french_homonyms(df, model, start, end)


if __name__ == "__main__":
    asyncio.run(main())