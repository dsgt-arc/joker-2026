import configparser
import os
from pathlib import Path

config = configparser.ConfigParser()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.ini"
CONFIG_PATH = Path(os.environ.get("JOKER_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))

if not config.read(CONFIG_PATH):
    raise FileNotFoundError(f"Could not read config file at {CONFIG_PATH}")

for section in ("model", "path", "dir"):
    if section not in config:
        raise KeyError(f"Missing [{section}] section in config file at {CONFIG_PATH}")

openrouter_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_key:
    raise EnvironmentError("Missing OPENROUTER_API_KEY")

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
# Short aliases are used for CLI args and filesystem directories.
# Values are OpenRouter model IDs.

gemini = config["model"].get("gemini", "google/gemini-3-flash-preview")
gemini_pro = config["model"].get("gemini_pro", "google/gemini-3.1-pro-preview")
gpt = config["model"].get("gpt", "openai/gpt-5.5")
claude = config["model"].get("claude", "anthropic/claude-sonnet-4.6")
deepseek = config["model"].get("deepseek", "deepseek/deepseek-v4-pro")
qwen = config["model"].get("qwen", "qwen/qwen3-max")

o3 = config["model"].get("o3", "")
o4 = config["model"].get("o4", "")
mistral = config["model"].get("mistral", "")
opus = config["model"].get("opus", "")

camembert = config["model"].get("camembert", "Lajavaness/sentence-camembert-large")
bilingual = config["model"].get("bilingual", "Lajavaness/bilingual-embedding-large")

MODEL_ALIASES = {
    "gemini": gemini,
    "gemini_pro": gemini_pro,
    "gpt": gpt,
    "claude": claude,
    "deepseek": deepseek,
    "qwen": qwen,
    "o3": o3,
    "o4": o4,
    "mistral": mistral,
    "opus": opus,
    "camembert": camembert,
    "bilingual": bilingual,
}

GENERATOR_MODEL_ALIASES = [
    "gemini",
    "gemini_pro",
    "claude",
    "gpt",
    "deepseek",
    "qwen",
]

GENERATOR_MODELS = [
    MODEL_ALIASES[alias]
    for alias in GENERATOR_MODEL_ALIASES
    if MODEL_ALIASES.get(alias)
]

JUDGE_MODEL_ALIASES = [
    "gpt",
    "claude",
]

JUDGE_MODELS = [
    MODEL_ALIASES[alias]
    for alias in JUDGE_MODEL_ALIASES
    if MODEL_ALIASES.get(alias)
]

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

translation_path = config["path"]["translation"]
translation_en_path = config["path"]["translation_en"]
translation_fr_path = config["path"]["translation_fr"]

location_en_input_path = config["path"]["location_en_input"]
location_fr_input_path = config["path"]["location_fr_input"]
location_en_qrels_path = config["path"]["location_en_qrels"]
location_fr_qrels_path = config["path"]["location_fr_qrels"]
location_manual_path = config["path"]["location_manual"]

cleaned_en_path = config["path"]["cleaned_en"]
cleaned_fr_path = config["path"]["cleaned_fr"]
combined_en_path = config["path"]["combined_en"]
combined_fr_path = config["path"]["combined_fr"]

identification_gpt_4o_path = config["path"]["identification_gpt_4o"]
refinement_gpt_4o_path = config["path"]["refinement_gpt_4o"]
fasttext_en_path = config["path"]["fasttext_en"]
fasttext_fr_path = config["path"]["fasttext_fr"]
fasttext_align_en_path = config["path"]["fasttext_align_en"]
fasttext_align_fr_path = config["path"]["fasttext_align_fr"]

contrastive_path = config["path"]["contrastive"]
phonetic_phrases_path = config["path"]["phonetic_phrases"]

phonetic_items_path = config["path"]["phonetic_items"]
phonetic_embeddings_path = config["path"]["phonetic_embeddings"]
phonetic_index_path = config["path"]["phonetic_index"]
phonetic_model_path = config["path"]["phonetic_model"]

# ------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------

identify_dir = config["dir"]["identify"]
translate_dir = config["dir"]["translate"]
similarity_dir = config["dir"]["similarity"]
homonym_dir = config["dir"]["homonym"]
generate_dir = config["dir"]["generate"]
contrastive_baseline_dir = config["dir"]["contrastive_baseline"]
contrastive_dir = config["dir"]["contrastive"]
phonetic_dir = config["dir"]["phonetic"]
retrieval_dir = config["dir"]["retrieval"]
