from data import load, save
from utils import get_response_not_json
from config import identification_gpt_4o_path, refinement_gpt_4o_path
import pandas as pd
import re
import os


evaluators = {
    "authenticity": {
        "instructions": """You are fluent in both English and French, with a deep understanding of humor in both languages. Your task is to compare a translated pun to 
        its original version and assess to what extent the translated pun reads like standard, well-edited language, such that the pun would be understood by a native speaker of 
        the French language.

Rate the translation 0 - 4 using one of the following categories:
    0 : Not at all likely
    1 : Not Very Likely
    2 : Somewhat Likely
    3 : Very Likely
    4 : Extremely Likely

Evaluate the pun based on these criteria and provide a justification in english for your rating. Provide your answer in the following format:

Rating: <number from 0 - 4>
Justification: <very concise explanation of key issues>"""
    },
    "humor": {
        "instructions": """You are a native French speaker with a sharp sense of humor. Your task is to evaluate a French pun on its own merits — not by comparing it to the English original.

Rate how funny and enjoyable the pun is to a native French speaker, from 0 to 3:
    3: Very funny — the wordplay lands immediately and provokes genuine amusement
    2: Mildly funny — the wordplay is clear and competent, but not particularly clever or surprising
    1: Weak — the pun is detectable but feels forced, flat, or unsatisfying
    0: Not funny — the wordplay is too obscure, broken, or awkward to register as humorous

Provide your answer in the following format:

Rating: <number from 0 - 3>
Justification: <very concise explanation of key issues>"""
    },
    "cleverness": {
        "instructions": """You are fluent in both English and French, with a deep understanding of wordplay in both languages. Your task is to evaluate how clever and elegant a French pun is — specifically, how well its double meaning works.

Rate the cleverness of the pun from 0 to 3:
    3: Very clever — the two meanings interlock naturally and the double meaning is satisfying and elegant
    2: Somewhat clever — the double meaning is present and intentional, but the connection feels slightly strained or coincidental
    1: Weak — the double meaning barely works, requires significant effort to see, or feels contrived
    0: Not clever — there is no functional double meaning, or the attempt at wordplay fails entirely

Provide your answer in the following format:

Rating: <number from 0 - 3>
Justification: <very concise explanation of key issues>"""
    },
    "recognizability": {
        "instructions": """You are a native French speaker. Your task is to evaluate how immediately and effortlessly a French pun is understood — not whether it is a good translation, but whether a native speaker would instantly recognize and appreciate the joke.

Rate the recognizability of the pun from 0 to 3:
    3: Immediately obvious — a native speaker would get the joke on first read with no effort
    2: Mostly clear — a native speaker would get it quickly, but may need a brief moment of reflection
    1: Too subtle — the pun is there but most native speakers would likely miss it or not find it worth the effort
    0: Opaque — the wordplay would not register as a joke to a typical native French speaker

Provide your answer in the following format:

Rating: <number from 0 - 3>
Justification: <very concise explanation of key issues>"""
    },
    "creativity": {
        "instructions": """You are fluent in both English and French, with a deep understanding of humor and wordplay. Your task is to evaluate how creative and inventive a French pun is — does it feel like a fresh, surprising piece of wordplay, or a mechanical/predictable substitution?

Rate the creativity of the pun from 0 to 3:
    3: Very creative — the wordplay feels inventive, fresh, and surprising in a satisfying way
    2: Somewhat creative — the wordplay is competent and not entirely predictable, but not particularly inventive
    1: Formulaic — the pun feels mechanical or like an obvious, first-pass solution
    0: Not creative — the wordplay is either nonexistent or purely substitutional with no inventive quality

Provide your answer in the following format:

Rating: <number from 0 - 3>
Justification: <very concise explanation of key issues>"""
    },
}

max_scores = {
    "authenticity": 4,
    "humor": 3,
    "cleverness": 3,
    "recognizability": 3,
    "creativity": 3,
}


def parse_evaluator_response(text):
    # Try to extract rating
    rating_match = re.search(r"Rating:\s*(\d+)", text)
    rating = int(rating_match.group(1)) if rating_match else None

    # Extract justification
    justification_match = re.search(r"Justification:\s*(.*)", text, re.DOTALL)
    justification = (
        justification_match.group(1).strip()
        if justification_match
        else "No justification"
    )
    print(f"Rating: {rating}, Justification: {justification}")
    return {"rating": rating, "justification": justification}


def aggregate_evaluations(evaluator_responses):
    thresholds = {
        "authenticity": 3,
        "humor": 2,
        "cleverness": 2,
        "recognizability": 2,
        "creativity": 2,
    }

    # Parse responses
    evaluations = {
        key: parse_evaluator_response(resp) for key, resp in evaluator_responses.items()
    }

    # Extract scores
    scores = {
        key: eval["rating"]
        for key, eval in evaluations.items()
        if eval["rating"] is not None
    }

    # Check if we received scores from all evaluators
    if len(scores) < len(thresholds):
        print("Missing scores from some evaluators. Re-evaluation.")
        return "refine", evaluations

    # Determine if all scores meet thresholds
    if all(scores[key] >= thresholds[key] for key in thresholds):
        print("Translation is good enough. Accepting it.")
        return "accept", evaluations

    print("Translation needs improvement.")
    return "refine", evaluations


# The input DataFrame should have the following columns:
# - "text_clean": The original English pun text.
# - "initial_translation": The initial French translation of the pun.
# - "id_en": An identifier for the English pun.
# - "is_pun": A binary indicator (1 for pun, 0 for non-pun).
# - change the dataframe this is loading from in config.ini at identification_gpt_4o_path
# - This saves to refinement_gpt_4o_path
# - Sentences identified as non puns and were not refined will have a "0" in the "iteration" column.


def refine_translations(df, model):
    checkpoint_path = "refined_translations_progress.csv"
    done_ids = set()
    if os.path.exists(checkpoint_path):
        done_df = pd.read_csv(checkpoint_path)
        done_ids = set(done_df["id_en"])

    for idx, row in df.iterrows():
        english_pun = row["text_clean"]
        current_translation = row["generated_pun"]
        id_en = row["id_en"]
        is_pun = row["is_pun"]
        max_iterations = 5
        iteration = 0

        if id_en in done_ids:
            continue

        if is_pun == 1:
            iteration = 1
            best_score = -1
            best_translation = current_translation
            best_iteration = 0

            while iteration < max_iterations + 1:
                print(
                    f"\n--------------Iteration {iteration} for Row {idx}---------------"
                )
                print("\nCURRENT TRANSLATION: ", current_translation)

                evaluator_responses = {}
                for key, evaluator in evaluators.items():
                    input_text = f"{evaluator['instructions']}\n\n{english_pun}\n\n{current_translation}"
                    response = get_response_not_json(input_text, model)
                    evaluator_responses[key] = response

                # Get decision and parsed evaluations
                decision, evaluations = aggregate_evaluations(evaluator_responses)

                # Compute raw average score (no normalization)
                scores = [
                    eval["rating"]
                    for eval in evaluations.values()
                    if eval["rating"] is not None
                ]
                avg_score = sum(scores) / len(scores) if scores else 0

                if avg_score > best_score:
                    best_score = avg_score
                    best_translation = current_translation
                    best_iteration = iteration

                if decision in ["accept", "minor_fix"]:
                    print(
                        f"Final translation after {iteration} iteration(s): {current_translation}"
                    )
                    break

                feedback_text = "\n\n".join(
                    [
                        f"{key.upper()} FEEDBACK:\n{resp}"
                        for key, resp in evaluator_responses.items()
                    ]
                )

                refinement_prompt = f"""
                The following pun was translated into French but did not meet all quality standards.
                
                English Pun: {english_pun}
                Current French Translation: {current_translation}
                
                Here is feedback from evaluators:
                {feedback_text}
                
                Improve the translation based on this feedback.
                
                Provide an improved translation. Only provide the French translation in your response:
                """

                current_translation = get_response_not_json(refinement_prompt, model)
                iteration += 1

            print(
                f"Best translation chosen after refinement: {best_translation} (Score: {best_score})"
            )
            df.at[idx, "id_en"] = id_en
            df.at[idx, "final_translation"] = best_translation
            df.at[idx, "iteration"] = best_iteration
            df.at[idx, "best_score"] = best_score

            df.loc[[idx]].to_csv(
                checkpoint_path,
                mode="a",
                header=not os.path.exists(checkpoint_path),
                index=False,
            )
    return df


if __name__ == "__main__":
    model = "o4"

    df = load(identification_gpt_4o_path)
    df = refine_translations(df, model)
    save(df, refinement_gpt_4o_path)
