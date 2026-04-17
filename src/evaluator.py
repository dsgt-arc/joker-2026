import ast
import sys

import numpy as np
import pandas as pd
from config import contrastive_dir, generate_dir, identify_dir, translate_dir
from data import combine_en, load, load_all, save
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import get_response

from sentence_transformers import SentenceTransformer, util
import torch

MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32

print("Using device:", DEVICE)


def evaluate_pun_location(df):
  y_true = df['manual_location'].str.lower()
  y_pred = df['pun_word'].str.lower()

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  print('pun_location')
  print('accuracy:', accuracy)
  print('precision:', precision)
  print('recall:', recall)
  print('f1-score:', f1, '\n')


def evaluate_pun_type(df):
  y_true = df['manual_type'].str.lower()
  y_pred = df['pun_type'].str.lower()

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  print('pun_type')
  print('accuracy:', accuracy)
  print('precision:', precision)
  print('recall:', recall)
  print('f1-score:', f1, '\n')


def evaluate_alternative_words(df, prompt_llm):
  def apply(row):
    manual_alternative = row['manual_alternative'].lower()
    generated_alternative = row['pun_alternative'].lower()

    print(row.name, manual_alternative, generated_alternative)
    if manual_alternative == generated_alternative:
      print('{"bool": 1}')
      return pd.Series({"bool": 1})

    schema = '{ "bool": 0 or 1 }'
    prompt = f"""
      Does the semantic range of "{generated_alternative}" overlap with the semantic range of "{manual_alternative}"? If yes return 1, else return 0.
      Return the output as a json using this schema: {schema}
    """
    return get_response(prompt, 'gpt-4o')

  if prompt_llm:
    df['evaluated_alternative'] = df.apply(apply, axis=1)
    save(df, identification_gpt_4o_path)

  loaded_df = load(identification_gpt_4o_path)
  y_true = loaded_df['manual_alternative'].str.lower()
  y_pred = loaded_df.apply(lambda row: row['manual_alternative'].lower() if row['evaluated_alternative'] else 'false', axis=1)

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  print('pun_alternative')
  print('accuracy:', accuracy)
  print('precision:', precision)
  print('recall:', recall)
  print('f1-score:', f1, '\n')


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


def encode_list(model, texts):
  texts = safe_list(texts)
  if len(texts) == 0:
    return None

  texts = [f"Represent this sentence for similarity: {t}" for t in texts]
  return model.encode(
    texts,
    batch_size=BATCH_SIZE,
    convert_to_tensor=True,
    normalize_embeddings=True,
  )


def pairwise_mean_cos(a, b):
  if a is None or b is None or len(a) == 0 or len(b) == 0:
    return np.nan
  sims = util.cos_sim(a, b)
  return sims.mean().item()


def pairwise_max_cos(a, b):
  if a is None or b is None or len(a) == 0 or len(b) == 0:
    return np.nan
  sims = util.cos_sim(a, b)
  return sims.max().item()


def mean_topk_cos(a, b, k=2):
  if a is None or b is None or len(a) == 0 or len(b) == 0:
    return np.nan
  sims = util.cos_sim(a, b).flatten()
  k = min(k, sims.numel())
  topk = torch.topk(sims, k=k).values
  return topk.mean().item()


def evaluate_translations(df):
  model = SentenceTransformer(MODEL_NAME, device=DEVICE)

  rows = []

  for _, row in df.iterrows():
    if row.get('pun_word_bt') == 'ERROR':
      continue

    first_en = encode_list(model, row['first_meaning'])
    second_en = encode_list(model, row['second_meaning'])
    first_bt = encode_list(model, row['first_meaning_bt'])
    second_bt = encode_list(model, row['second_meaning_bt'])

    if first_en is None or second_en is None or first_bt is None or second_bt is None:
      continue

    # Main discrimination scores
    first_to_first = mean_topk_cos(first_en, first_bt, k=2)
    first_to_second = mean_topk_cos(first_en, second_bt, k=2)
    second_to_second = mean_topk_cos(second_en, second_bt, k=2)
    second_to_first = mean_topk_cos(second_en, first_bt, k=2)

    # Gaps: positive is good
    first_gap = first_to_first - first_to_second
    second_gap = second_to_second - second_to_first
    avg_gap = np.nanmean([first_gap, second_gap])

    # Win indicators
    first_win = int(first_to_first > first_to_second)
    second_win = int(second_to_second > second_to_first)
    row_correct = int(first_win and second_win)

    # Optional within-language separation sanity checks
    en_sep = mean_topk_cos(first_en, second_en, k=2)
    bt_sep = mean_topk_cos(first_bt, second_bt, k=2)

    rows.append({
      'id_en': row.get('id_en', ''),
      'pun_word': row.get('pun_word', ''),
      'first_to_first': first_to_first,
      'first_to_second': first_to_second,
      'second_to_second': second_to_second,
      'second_to_first': second_to_first,
      'first_gap': first_gap,
      'second_gap': second_gap,
      'avg_gap': avg_gap,
      'first_win': first_win,
      'second_win': second_win,
      'row_correct': row_correct,
      'en_separation': en_sep,
      'bt_separation': bt_sep,
    })

  result_df = pd.DataFrame(rows)

  if len(result_df) == 0:
    print('No valid rows to evaluate.')
    return result_df

  print('row count', len(result_df))

  print('\n--- discrimination accuracy ---')
  print('first sense win rate', result_df['first_win'].mean())
  print('second sense win rate', result_df['second_win'].mean())
  print('row fully correct rate', result_df['row_correct'].mean())

  print('\n--- discrimination gaps (higher is better) ---')
  print('first gap mean', result_df['first_gap'].mean())
  print('second gap mean', result_df['second_gap'].mean())
  print('avg gap mean', result_df['avg_gap'].mean())
  print('avg gap variance', result_df['avg_gap'].var())

  print('\n--- same-vs-cross similarities ---')
  print('first->first mean', result_df['first_to_first'].mean())
  print('first->second mean', result_df['first_to_second'].mean())
  print('second->second mean', result_df['second_to_second'].mean())
  print('second->first mean', result_df['second_to_first'].mean())

  print('\n--- separation sanity check ---')
  print('EN separation mean', result_df['en_separation'].mean())
  print('BT separation mean', result_df['bt_separation'].mean())

  hard_cases = result_df.sort_values('avg_gap').head(10)
  print('\n--- hardest rows ---')
  print(hard_cases[['id_en', 'pun_word', 'first_gap', 'second_gap', 'avg_gap']])

  return result_df


def evaluate_generations(df, context_df, eval_model, start=0, end=-1):
  def create_context_string(row):
    text = row['text_clean']
    target = row['target']
    prefix = 'Contains a pun: ' if target == 1 else 'Does not contain a pun: '
    return prefix + text

  pun_df = context_df[context_df['target'] == 1].sample(n=25)
  non_pun_df = context_df[context_df['target'] == 0].sample(n=25)
  context_df = pd.concat([pun_df, non_pun_df], axis=0)
  context_df['string'] = context_df.apply(create_context_string, axis=1)
  context = '\n'.join(context_df['string'].tolist())

  def apply(row):
    text = row['generated_pun']
    schema = '{ "is_pun": 0 or 1 }'
    prompt = f"""
      {context}
      Input: {text}
      If the input contains a pun return 1, else return 0, in a properly formatted json using this schema: {schema}
    """
    print(row.name, text)
    try:
      response = get_response(prompt, eval_model)
    except ValueError as e:
      print(f'Error: {e}')
      response = '{ "is_pun": "ERROR" }'
    return response

  chunk_size = 10
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  if end == -1:
    end = len(chunks)
  for i in range(start, end):
    chunks[i][['is_pun']] = chunks[i].apply(apply, axis=1)
    save(chunks[i], f'{contrastive_dir}baseline/{eval_model}/{model}/{i}.tsv')


# =========================
# MAIN (UNCHANGED)
# =========================

if __name__ == "__main__":
  task = sys.argv[1]
  model = sys.argv[2]
  eval_model = sys.argv[3] if len(sys.argv) > 3 else ''
  start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
  end = int(sys.argv[5]) if len(sys.argv) > 5 else -1

  if task == 'identify':
    df = load_all(f'{identify_dir}{model}/')
    save(df, f'{identify_dir}{model}.tsv')
    df = load(f'{identify_dir}{model}.tsv')
    df = df[df['manual_location'].str.len() > 0]
    print('row count', len(df))
    evaluate_pun_location(df)
    evaluate_pun_type(df)

  if task == 'translate':
    df = load_all(f'{translate_dir}{model}/')
    save(df, f'{translate_dir}{model}.tsv')
    df = load(f'{translate_dir}{model}.tsv')
    df = df[df['pun_word_bt'].str.len() > 0]
    print('row count', len(df))
    evaluate_translations(df)

  if task == 'generate':
    context_df = load(f'{contrastive_dir}dataset.csv')
    print('context count', len(context_df))

    df = load_all(f'{generate_dir}{model}/')
    save(df, f'{generate_dir}{model}.tsv')
    print('generate count', len(df))
    evaluate_generations(df, context_df, eval_model, start, end)

  if task == 'gen_count':
    df = load_all(f'{contrastive_dir}baseline/o4/{model}/')
    save(df, f'{contrastive_dir}baseline/o4/{model}.tsv')
    print('eval_model=o4 - row count', len(df))
    print(df['is_pun'].value_counts(normalize=True))

    df = load_all(f'{contrastive_dir}baseline/gemini/{model}/')
    save(df, f'{contrastive_dir}baseline/gemini/{model}.tsv')
    print('\neval_model=gemini - row count', len(df))
    print(df['is_pun'].value_counts(normalize=True))
