# Changes from original get_cosine_similarity code to the updated version below:

''''

**1. From raw difference to a composite Low score**

The original code's primary output was `first_similarity_diff` and `second_similarity_diff` — simply subtracting the French cosine similarity from the English one. 
This is a thin signal because it treats both meanings symmetrically and tells you nothing about whether the pun is actually working as a pun. 
A translation could score well on one meaning and zero on the other and still look fine by the diff metric.

The new `compute_low_score` replaces this as the headline metric. It's grounded in Low's core insight: a good pun translation isn't one that's close to *either* meaning, 
it's one that straddles *both*. So the score weights `min(sim1, sim2)` — the balanced signal — most heavily, with `max(sim1, sim2)` as a secondary floor. 
The raw diffs are still computed and saved so nothing downstream breaks, but the Low score is the more theoretically meaningful output.

---

**2. Relation type bonus**

The original code ignored how the French pun candidate was derived — whether it was a direct translation, a synonym, or a homophone. 
The notebook's `_score_low_candidate` made clear that this matters: homophones are structurally the strongest pun mechanism 
(Low's whole polygon framework is built around phonetic proximity), followed by synonyms, followed by direct translations. 

The `RELATION_BONUS` map formalises this, feeding into the composite score via the `relation` weight.

---

**3. Same-as-base penalty**

A small but principled addition from the notebook. If the French pun word is just the literal translation of the English pun word, 
that's the trivial non-solution — it means no creative work has been done to find a pun in the target language. 

The 0.10 penalty discourages the scorer from rewarding this case, which keeps the metric honest.

---

**4. Pattern-aware weights via `PUN_PATTERN_WEIGHTS`**

This is the change driven by the pun structure analysis. The original code (and even the first rewrite) used fixed weights regardless of 
what kind of pun was being evaluated. But the four pattern types differ in where the ambiguity lives:

- In a **word pivot**, the pun word embedding is a reliable carrier of both meanings, so the original weight balance holds.

- In a **phrase pivot**, the ambiguity is spread across a phrase rather than a single token, so the strong meaning signal is slightly less trusted.

- In a **structure-based** pun, the pun word alone is a weak signal because the twist depends on sentence-level grammar. Here the strong meaning 
weight increases slightly — you want at least one meaning well-covered even if balance is harder to achieve.

- In a **full reinterpretation** pun, the entire setup leads you one way before the twist. Balance between meanings is most critical here because 
both A and B need to be plausible throughout, and the relation type bonus matters least since the work is being done at the narrative level, not the token level.

The `unknown` fallback preserves the original weights exactly, so any rows without a `pun_pattern` column are unaffected and the function remains backwards compatible.

---

**The thread connecting all of it**

Every change is ultimately doing the same thing: moving the evaluation metric closer to what Low actually describes as a successful pun translation. 

Low's argument is that the goal is to preserve the *effect* of the joke — the expectation/twist structure — not the surface form. 

The original cosine similarity code measured surface proximity. The rewritten code measures how well the A→B twist is preserved, weighted by the structural 
type of pun and the phonetic/semantic path used to get there.

''''

# Re-written get_cosine_similarity

def get_cosine_similarity(df, model, start=0, end=-1):

    RELATION_BONUS = {
        "homophone": 0.90,
        "synonym":   0.55,
        "direct":    0.35,
    }

    # Per Low's polygon logic, pattern type shifts how much we weight
    # balance vs. strength vs. relation type.
    # - Word pivot: single token carries all ambiguity, balance matters most
    # - Phrase pivot: distributed across a phrase, balance still key
    # - Structure-based: the pun word is a weaker carrier, reduce its weight,
    #                    lean more on strong_meaning as a floor signal
    # - Full reinterpretation: pun word embedding is least reliable,
    #                          balance is critical, relation bonus matters less
    PUN_PATTERN_WEIGHTS = {
        "word_pivot":           {"balanced": 0.50, "strong": 0.30, "relation": 0.25},
        "phrase_pivot":         {"balanced": 0.50, "strong": 0.25, "relation": 0.25},
        "structure_based":      {"balanced": 0.40, "strong": 0.35, "relation": 0.25},
        "full_reinterpretation":{"balanced": 0.55, "strong": 0.25, "relation": 0.20},
        "unknown":              {"balanced": 0.50, "strong": 0.30, "relation": 0.25},
    }

    def compute_low_score(sim1, sim2, pun_type, relation_type, pun_word, base):
        weights = PUN_PATTERN_WEIGHTS.get(pun_type, PUN_PATTERN_WEIGHTS["unknown"])

        balanced_meaning     = min(sim1, sim2)
        strong_meaning       = max(sim1, sim2)
        relation_strength    = RELATION_BONUS.get(relation_type, 0.35)
        same_as_base_penalty = 0.10 if pun_word == base else 0.0

        return (
            weights["balanced"]  * balanced_meaning
          + weights["strong"]    * strong_meaning
          + weights["relation"]  * relation_strength
          - same_as_base_penalty
        )

    def apply(row, st_model):
        pun_word_embedding_en = st_model.encode(
            [row['pun_word']], convert_to_tensor=True)
        first_meaning_embedding_en = torch.mean(
            st_model.encode(ast.literal_eval(row['first_meaning']), convert_to_tensor=True),
            dim=0, keepdim=True)
        second_meaning_embedding_en = torch.mean(
            st_model.encode(ast.literal_eval(row['second_meaning']), convert_to_tensor=True),
            dim=0, keepdim=True)

        pun_word_embedding_fr = st_model.encode(
            [row['pun_word_fr']], convert_to_tensor=True)
        first_meaning_embedding_fr = torch.mean(
            st_model.encode(ast.literal_eval(row['first_meaning_fr']), convert_to_tensor=True),
            dim=0, keepdim=True)
        second_meaning_embedding_fr = torch.mean(
            st_model.encode(ast.literal_eval(row['second_meaning_fr']), convert_to_tensor=True),
            dim=0, keepdim=True)

        # Raw cosine similarities
        first_similarity_en  = util.cos_sim(pun_word_embedding_en, first_meaning_embedding_en).item()
        second_similarity_en = util.cos_sim(pun_word_embedding_en, second_meaning_embedding_en).item()
        first_similarity_fr  = util.cos_sim(pun_word_embedding_fr, first_meaning_embedding_fr).item()
        second_similarity_fr = util.cos_sim(pun_word_embedding_fr, second_meaning_embedding_fr).item()

        first_similarity_diff  = first_similarity_en - first_similarity_fr
        second_similarity_diff = second_similarity_en - second_similarity_fr

        # Pattern type and relation type drive the Low score weights
        pun_pattern   = row.get('pun_pattern', 'unknown')
        relation_type = row.get('pun_type', 'direct')
        base_fr       = row.get('base_fr', row['pun_word_fr'])

        low_score_en = compute_low_score(
            first_similarity_en, second_similarity_en,
            pun_pattern, relation_type,
            row['pun_word'], row['pun_word'],  # EN word is its own base
        )
        low_score_fr = compute_low_score(
            first_similarity_fr, second_similarity_fr,
            pun_pattern, relation_type,
            row['pun_word_fr'], base_fr,
        )
        low_score_diff = low_score_en - low_score_fr

        print(row.name, row['pun_word'], row['pun_word_fr'], pun_pattern, relation_type)
        print('first  en', first_similarity_en,  'fr', first_similarity_fr,  'diff', first_similarity_diff)
        print('second en', second_similarity_en, 'fr', second_similarity_fr, 'diff', second_similarity_diff)
        print('low    en', low_score_en, 'fr', low_score_fr, 'diff', low_score_diff)

        return pd.Series({
            'first_similarity_en':    first_similarity_en,
            'second_similarity_en':   second_similarity_en,
            'first_similarity_fr':    first_similarity_fr,
            'second_similarity_fr':   second_similarity_fr,
            'first_similarity_diff':  first_similarity_diff,
            'second_similarity_diff': second_similarity_diff,
            'low_score_en':           low_score_en,
            'low_score_fr':           low_score_fr,
            'low_score_diff':         low_score_diff,
        })

    st_model   = get_model(model)
    chunk_size = 10
    chunks     = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)

    for i in range(start, end):
        current_df = chunks[i]
        current_df[['first_similarity_en', 'second_similarity_en',
                     'first_similarity_fr', 'second_similarity_fr',
                     'first_similarity_diff', 'second_similarity_diff',
                     'low_score_en', 'low_score_fr', 'low_score_diff']] = \
            current_df.apply(apply, axis=1, args=(st_model,))
        save(current_df, f'{similarity_dir}{model}/{i}.tsv')