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

gemini = config["model"]["gemini"]
gemini_pro = config["model"]["gemini_pro"]

gpt = config["model"]["gpt"]
claude = config["model"]["claude"]
deepseek = config["model"]["deepseek"]
qwen = config["model"]["qwen"]

o3 = config["model"].get("o3", "")
o4 = config["model"].get("o4", "")
mistral = config["model"].get("mistral", "")
opus = config["model"].get("opus", "")

camembert = config["model"].get("camembert", "")
bilingual = config["model"].get("bilingual", "")

GENERATOR_MODELS = [
    claude,
    gpt,
    gemini_pro,
    deepseek,
    qwen,
]

JUDGE_MODELS = [
    gpt,
    claude,
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

contrastive_path = config["path"]["contrastive"]

identification_gpt_4o_path = config["path"]["identification_gpt_4o"]
refinement_gpt_4o_path = config["path"]["refinement_gpt_4o"]

fasttext_en_path = config["path"]["fasttext_en"]
fasttext_fr_path = config["path"]["fasttext_fr"]

fasttext_align_en_path = config["path"]["fasttext_align_en"]
fasttext_align_fr_path = config["path"]["fasttext_align_fr"]

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

retrieval_dir = config["dir"]["retrieval"]

similarity_dir = config["dir"]["similarity"]
homonym_dir = config["dir"]["homonym"]

generate_dir = config["dir"]["generate"]

contrastive_baseline_dir = config["dir"]["contrastive_baseline"]
contrastive_dir = config["dir"]["contrastive"]

phonetic_dir = config["dir"]["phonetic"]