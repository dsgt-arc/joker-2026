import csv
import glob
import os
import pandas as pd
import re
from config import cleaned_en_path, cleaned_fr_path, combined_en_path, combined_fr_path, location_manual_path, location_fr_input_path, location_fr_qrels_path, translation_path
import warnings
warnings.filterwarnings('ignore')

# Dataset-specific cleaner for joker_task2_en_fr_2026_test.json.
# Goal: maximize LLM comprehension of the pun, not preserve exact original formatting.

SPEECH_VERBS = (
  r"(?:said|asked|replied|reported|cried|yelled|spouted|boasted|"
  r"admitted|chimed|mourned|bellowed|snapped|nagged|pleaded|"
  r"projected|allowed|guessed|whispered|shouted|queried|murmured|"
  r"read|answered|retorted|remarked|professed|assented|delivered|"
  r"advanced|called|blustered|clamored|ventured|moped|refused|"
  r"bickered|bristled|pondered|coached|upheld|snorted)"
)

SOCIAL_TAGS = [
  'dadjoke', 'dadjokes', 'funny', 'pun', 'puns', 'joke', 'jokes',
  'wordplay', 'humour', 'humor', 'lol', 'oneliner', 'funquestion',
  'instagramposts', 'instagramreels', 'hipster', 'genz', 'comedy',
  'jossofalva'
]

# Manual overrides for rows where a deterministic corpus-wide rule is not the clearest option.
# These are intentionally specific to THIS dataset.
MANUAL_TEXT_CLEAN = {
  24: '"" he said blankly.',
  25: '"3.14159265," he said piously.',
  112: '"Waiter, there\'s a fly in my soup!" "Force of habit, sir, the chef used to be a tailor."',
  124: '"Because" is a word to the whys.',
  202: 'A: Mr. Edgeworth, please find out more about your witness, Ms. Windy... what was her name? B: Something "Oldbag," Your Honor. A: Then the prosecution will look further into this Oldbag before we continue!',
  1339: 'I didn\'t "hear" the time, I "saw" it!',
  3506: "Today I saw an ad that said \"radio for sale, $1, volume is stuck at max level\". I thought: \"I just can't turn that down.\"",

  # Final row-specific repairs found by full validation.
  212: 'A bacteria walked into a bar and the bartender said, "We don\'t serve bacteria in this place." The bacteria said, "But I work here, I\'m staph."',
  213: 'A bacteria walked into a bar and the bartender said, "We don\'t serve bacteria in this place." The bacteria said, "But I work here, I\'m staph."',
  364: 'A man given a watch at his retirement said "it\'s about time".',
  406: 'A politician who had been an astronomer was always saying "no comet".',
  569: 'An optometrist told his patient: "It appears your vision is improving!" "Really?" replied the patient. Must be the luck of the iris.',
  780: 'Curiouser and curiouser!',
  960: "Faith doesn't fall apart at the seems",
  1105: 'He admired my huge success in the steel industry but I told him "Anybody can get to the top in anything if they have the desire to. There\'s no need to be in ore."',
  1860: "It wasn't the apple on the tree, but the pair beneath it.",
  2508: 'One palm tree said to another "let\'s have a date."',
  3473: "Time flies when you're following the son",
  3519: "Try our Sundays, they're better than Baskin Robbins",
  3770: 'When asked by a passenger how high he would get, the pilot replied, "I don\'t do drugs."',
  3771: 'When asked by a passenger how high he would get, the pilot replied, "I don\'t do drugs."',

  # Dataset-specific quote repairs for compressed quoted words/phrases.
  233: 'A book called "Current Trends in Wiring your House" turned out to be a shocking failure.',
  235: 'A bowling team was called "lightning" because they had so many strikes.',
  351: 'A little boy called his father who made balloons "pop".',
  397: 'A photographer taking pictures of golfers says "watch the birdie".',
  636: "Audubon said he'd have to wing it.",
  785: "Darwin said he'd have to see what evolved.",
  1107: 'He always called his girlfriend his "luck". Until he decided to push his luck.',
  2509: "One palm tree said to another \"let's have a date.\"",
  2863: "Stealing someone's coffee is called \"mugging\".",
  3279: 'The salt said "hi" to the pepper. It was seasonings greetings.',
  3383: 'There was a sign on the lawn at a drug re-hab center that said "Keep off the Grass".',
  3541: 'Two lovers who had been apart for some time were reunited on a foggy day. One whispered to the other "I mist you".',
  3976: "Wilbur Wright said he'd take a flier on it.",
  # Final manual repairs from full output evaluation.
  566: 'An old lady once asked the dispatcher of a local trucking company if they could ship an antique mirror to her sister in Toronto. The dispatcher says, "I don\'t know madam, I\'d have to look into it first."',
  727: '"Change the channel," she said remotely.',
  1273: '"I agree with you wholeheartedly," said the artichoke grower.',
  1374: '"I got lost in the streets of Paris," he said ruefully.',
  1731: '"If you breathe heavily on the map, it will reveal topography," he said, with a sigh of relief.',
  1732: '"If you breathe heavily on the map, it will reveal topography," he said, with a sigh of relief.',
  2065: 'My friend said, "There\'s a lot of gold in those hills." I replied, "That\'s a load of bullion."',
  2171: 'My shrink assures me that my obsession with the formalization of puns is just a "phrase I\'m going through".',
  2908: '"That\'s the reason they\'re called lessons," the Gryphon remarked: "because they lessen from day to day."',
  3177: 'The mathematics professor, lamenting his students\' lackadaisical approach to trigonometry, sighed, "It\'s a sine of the times."',
  3329: 'The triglyph commented, "It\'s friezing in here."',
  3330: 'The triglyph commented, "It\'s friezing in here."',
  3655: 'What did the minister say to the underdressed layman? "No shoes, no shirt, no service."',
  3884: 'When your internet provider goes bankrupt it\'s a "net loss".',
  # Final pass repairs from evaluation of v9 output.
  10: '"I think it be thine, indeed; for thou liest in\'t" "You lie out on\'t, sir, and therefore \'tis not yours. For my part, I do not \'lie\' in\'t, and yet it is mine." Thou dost \'lie\' in\'t, to be in\'t and say it is thine. "\'Tis for the dead, not for the quick; therefore thou liest." "\'Tis a \'quick\' lie, sir; \'twill away again from me to you."',
  17: '"So what do you do?" "Nothing fancy, just been able to put the food on the table." "Okay, but what\'s your profession?" "Waiter."',
  873: 'Doctor, Doctor, this ointment you gave me makes my arm smart. Then rub some on your head? Next.',
  1395: "I have a rumour about peanut butter. But I don't want to spread it.",
  1497: 'I sold my vacuum cleaner today. All it was doing was collecting dust.',
  2076: "My mate who's an origami teacher, has quit her job. Apparently... 1. There was too much paperwork. 2. She kept folding under pressure. 3... she just couldn't cut it...",
  2082: "My name is Bea. I'm in the honey business.",
  2249: 'Old cardiologists never die, they are just repulsed.',
  2294: 'Old gangsters never die, they just go to the underworld.',
  2311: 'Old hairdressers never die, they just braid away.',
  2394: 'Old shoe shine boys never die, they are just rebuffed.',
  2421: 'Old upholsterers never die, they just recover.',
  2500: 'One item contributed was a picture of a pretty, kimono-clad girl; it bore the inscription: "Maid in Japan".',
  2731: "She was only a Swimmer's daughter, but she knew every dive in town.",
  2737: "She was only a Wrestler's daughter, but you oughtta see her box.",
  # Capitalization/manual readability repairs for LLM pun extraction.
  # Keep caps when they are the pun cue or a true acronym; normalize source-format shouting/speaker labels.
  7: '"Before she had this fit" You never had fits, my dear, I think? Then the words don\'t fit you.',
  183: 'Mine is a long and a sad tale! - It is a long tail, certainly.',
  201: 'A: I knew that by not concealing myself, I would be putting pressure on the thief. B: (Looks like the thief was the one applying pressure... On your pidgeony head, that is.)',
  477: 'Algernon: Lane, you\'re a perfect pessimist. Lane: I do my best to give satisfaction, sir.',
  490: 'After 40 years, French TV show "Thalassa" no longer rocks the boat',
  598: 'Are your parents living? Jack: I have lost both my parents. Lady Bracknell: Both?... That seems like carelessness.',
  705: 'Cecily: Miss Prism says that all good looks are a snare. Algernon: They are a snare that every sensible man would like to be caught in.',
  706: 'Cecily: Well, I am really only eighteen, but I always admit to twenty when I go to evening parties. Lady Bra: You are perfectly right in making some slight alteration. Indeed, no woman should ever be quite accurate about her age. It looks so calculating...',
  707: 'Crossword. I guess it\'s an apt name. Those words make me cross!',
  782: 'Do you absolutely, dapsolutely want the solution?',
  1026: 'Gwendolen: If you are not too long, I will wait here for you all my life',
  1271: 'I love to be pet too! Sigh... It\'s all just a pupularity contest.',
  1396: 'I have always been of opinion that a man who desires to get married should know either everything or nothing. Which do you know? Jack: I know nothing, Lady Bracknell.',
  1608: 'I wonder if I shall fall right through the earth! How funny it\'ll seem to come out among the people that walk with their heads downward! The Antipathies, I think.',
  1892: 'Jack: Well, at any rate, that is better than being always overdressed as you are. Algernon: If I am occasionally a little over-dressed, I make up for it by being always immensely over-educated.',
  1916: 'Lady Bra: My nephew, you seem to be displaying signs of triviality. Jack: On the contrary, Aunt Augusta, I\'ve now realized for the first time in my life the vital Importance of Being Ernest.',
  1917: 'Lady Bracknell: Well, I must say, Algernon, that I think it is high time that Mr. Bunbury made up his mind whether he was going to live or to die.',
  1973: 'Miss Prism: That depends on the intellectual sympathies of the woman. Maturity can always be depended on. Ripeness can be trusted. Young women are green. [Dr. Chasuble starts.] I spoke horticulturally. My metaphor was drawn from fruits.',
  2004: 'Mess with me... and I\'ll make you cough it all up!',
  2432: 'Orange tiles are orange-scented. They will make you smell delicious!',
  2445: 'Old hippies never die, they just take a trip.',
  2447: 'Old nitpickers never die, they just feel lousy.',
  2603: 'Rockfish. Great Seafood. Not a lot of clams.',
  2630: 'So let\'s let bybones be bybones.',
  2884: 'This is Papyrus\'s hotful helpline!',
  2885: 'This puzzle! You figured it out so easily! That was very Papyrus of you.',
  2911: 'That, my dear Algy, is the whole truth pure and simple. Algernon: The truth is rarely pure and never simple. Modern life would be very tedious if it were either, and modern literature a complete impossibility!',
  3520: 'Tuborg. BEer yourself',
  3578: 'We\'re too refined for that greasehole.',
  3602: "We are KOA'mazing.",
  3646: 'What are your politics? Jack: Well, I am afraid I really have none. I am a Liberal Unionist.',
  3927: "Why can't a bicycle stand on its own without a wedge? Bro, can't you see, it's TWO-TIRED",
  3769: 'When asked by a Health Department official to describe the mess he saw on the slaughterhouse floor, the USDA inspector replied, "It was just offal."',
  4012: 'Yikes! The coolest way to take your vitamins!',
  # Additional nitpicky dataset-specific repairs for LLM pun extraction.
  4: '"A good [deed] is never lost. Character is [property], it is the noblest of possession."',
  22: '"Where is the thousand marks thou hadst of me?" "I have some marks of yours upon my pate, Some of my mistress\' marks upon my shoulders, But not a thousand marks between you both. If I should pay your Worship those again, Perchance you will not bear them patiently."',
  158: '"Sweeney Todd" is a good source of sheer terror.',
  168: '"We\'ve run out of lemons," she said bitterly.',
  172: '"" he said blankly.',
  179: "Call me a taxi. - Okay, you're a taxi.",
  760: 'Computers at breakfast food companies use serial I/O.',
  1293: '"I can\'t budge this huge box!" I exclaimed. "Of course not," the office supply warehouse clerk replied, "it\'s stationery."',
  1735: 'If you can\'t sell your home... "I KEN"',
  178: 'And the twinkling of the tea-. - The twinkling of the what? - It began with the tea. - Of course twinkling begins with a T!',
  2052: 'My family scoffed when I had a divine inspiration I\'d get rich selling my "Beets Brule," but after I made my first million, they had to admit I made quite a prophet.',
  2433: 'Occasionally, I hear an assertion followed by a hasty qualification, such as, "Jim graduated in 1995, if memory serves." I\'ll respond: "... or only stands and waits."',
  2758: 'Sign on a broken perfume bottle, "Out of odor."',
  3906: 'While baking, I dropped a stick of margarine on the wooden tile floor, and when my neighbor slipped and fell, I said "It must have been the parkay."',
  4055: '"The Predator," prey for life',
  4061: 'And they all said the RAF would never take off.',
  1862: "It's okay to borrow a book from the public library once in a while, but try not to overdue it.",
  4008: "Yesterday I accidentally swallowed some food coloring. The doctor says I'm okay, but I feel like I've dyed a little inside.",
  # Additional bracket/punctuation cleanups.
  702: "But under no circumstances can you chase after him. Eh, why's that? You don't want to get involved in any monkey business, right!?",
  1050: "Girlfriend: I'll have the Chef's salad. Me: Babe that's so rude, just order your own.",
  1973: 'Miss Prism: That depends on the intellectual sympathies of the woman. Maturity can always be depended on. Ripeness can be trusted. Young women are green. I spoke horticulturally. My metaphor was drawn from fruits.',
  2028: 'Ms. von Karma\'s logic is perfect. There is no way for you to poke a hole in it. (It looks like I\'ve found the "hole" I was looking for!)',
  3075: 'The first duck wouldn\'t go in the water. The other duck said "What are you, chicken?"',
  3508: 'Today, my friend asked "Can I have a book, Mark?" We are friends from 2 years and he still doesn\'t know my name is John.',
  3959: 'Why does everyone cry when they eat Mexican pizza? Because they are in the Café Teary Eye!',
  4054: 'And my ideal has always been to love some one of the name of Ernest. There is something in that name that inspires absolute confidence.',
}


def load(path):
  ext = os.path.splitext(path)[1]
  if ext == '.json':
    data = pd.read_json(path)
  elif ext == '.csv':
    data = pd.read_csv(path, dtype=str)
  elif ext == '.tsv':
    data = pd.read_csv(path, sep='\t', dtype=str)
  else:
    with open(path, 'r', encoding='utf-8') as file:
      data = file.read()

  print(f'Loaded {path}')
  if type(data) is pd.DataFrame:
    print(f'Row count:', len(data))
  return data.fillna('') if type(data) is pd.DataFrame else data


def load_all(path):
  files = glob.glob(os.path.join(path, "*.tsv"))
  files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else -1)

  if not files:
    raise FileNotFoundError(
      f"No TSV files found in {path}. "
      "Generator input is expected at ../data/processed/retrieval/gemini/ "
      "because retrieval was run with the gemini alias."
    )

  df_list = []

  for file in files:
    df = pd.read_csv(file, sep='\t', dtype=str)
    df_list.append(df)

  combined_df = pd.concat(df_list, ignore_index=True)
  print(f'Loaded all tsvs in {path}')
  print(f'Row count:', len(combined_df))
  return combined_df.fillna('')


def save(data, path):
  directory = os.path.dirname(path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)

  ext = os.path.splitext(path)[1]
  if ext == '.tsv':
    # Do NOT use QUOTE_NONE here. QUOTE_NONE + escapechar='\\' corrupts rows
    # that contain quote artifacts at the end of a field.
    data.to_csv(path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
  elif ext == '.csv':
    data.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
  else:
    if isinstance(data, pd.DataFrame):
      data.to_json(path, orient='records', force_ascii=False, indent=2)
    else:
      with open(path, 'w', encoding='utf-8') as file:
        file.write(data)

  print(f'Saved {path}')


def _lower_old_puns(sentence):
  words = sentence.split()
  if words and words[0] == "OLD":
    all_caps_sequence = True
    for i in range(1, len(words)):
      word = words[i]
      if len(word) <= 2:
        continue
      if not word.isupper():
        all_caps_sequence = False
      if word.isupper() and all_caps_sequence:
        words[i] = word.lower()
    words[0] = words[0].capitalize()
  return ' '.join(words)


def _repair_tom_before_neutralizing(sentence):
  # Keep this aggressive fix from your original script because it prevents
  # "said he" / "asked he" after Tom neutralization.
  sentence = re.sub(
    r"\bcried\s+Tom's\s+band\b",
    "Tom's band cried",
    sentence,
    flags=re.IGNORECASE
  )

  sentence = re.sub(
    rf"\b({SPEECH_VERBS})\s+Tom\b",
    lambda m: 'Tom ' + m.group(1).lower(),
    sentence,
    flags=re.IGNORECASE
  )

  # If the closing quote was lost before Tom, restore it.
  sentence = re.sub(
    rf'^"([^"]*?[,.!?])\s+Tom\s+({SPEECH_VERBS})\b',
    lambda m: f'"{m.group(1)}" Tom {m.group(2).lower()}',
    sentence,
    flags=re.IGNORECASE
  )

  return sentence


def _repair_after_neutralizing(sentence):
  # Safety net in case any "said he" pattern remains.
  sentence = re.sub(
    rf"\b({SPEECH_VERBS})\s+he\b",
    lambda m: 'he ' + m.group(1).lower(),
    sentence,
    flags=re.IGNORECASE
  )

  # Restore quote boundary after Tom -> he.
  sentence = re.sub(
    rf'^"([^"]*?[,.!?])\s+he\s+({SPEECH_VERBS})\b',
    lambda m: f'"{m.group(1)}" he {m.group(2).lower()}',
    sentence,
    flags=re.IGNORECASE
  )

  # Fix: !"he -> !" he
  sentence = re.sub(r'([,.!?])"(?=[A-Za-z])', r'\1" ', sentence)

  # If a row now has exactly one likely closing quote, add the opening quote.
  # This is intentionally aggressive for LLM comprehension.
  if sentence.count('"') % 2 == 1:
    if not sentence.startswith('"') and re.search(r'^[^"\n]{1,250}[,.!?]"', sentence):
      sentence = '"' + sentence
    elif sentence.startswith('"'):
      sentence = sentence + '"'

  return sentence



def _clean_hashtags_for_pun_recovery(sentence):
  """Dataset-specific hashtag handling.

  Social hashtags are removed. Inline lexical hashtags that mark likely pun words
  are preserved as bracketed emphasis, e.g. #deed -> [deed]. This improves the
  first LLM stage by keeping explicit pun-location cues without Twitter syntax.
  """
  # Remove known social/meta hashtags completely.
  for tag in SOCIAL_TAGS:
    sentence = re.sub(rf"\s*#\s*{tag}\b", "", sentence, flags=re.IGNORECASE)

  # Preserve remaining lexical hashtag markers as salience cues.
  sentence = re.sub(r"#([A-Za-z][A-Za-z0-9_]*)", r"[\1]", sentence)

  # Clean social framing left behind by removed meta hashtags.
  sentence = re.sub(r"(?i)^\s*Okay,\s*last\s+from\s+us!\s*", "", sentence)
  sentence = re.sub(r"(?i)\s*-\s*double\s*$", "", sentence)
  sentence = re.sub(r"\s{2,}", " ", sentence).strip()
  return sentence


def _remove_reaction_noise(sentence):
  """Remove trailing reaction/performance noise while preserving pun spellings.

  This intentionally does NOT remove 'HeHe' without punctuation, because in this
  dataset that can be the helium chemical-symbol pun.
  """
  # Remove standalone emoji reaction clutter.
  sentence = re.sub(r"[\U0001F300-\U0001FAFF]+", "", sentence)

  # Remove trailing reaction text like Hehe!, haha!, lol! but preserve HeHe.
  sentence = re.sub(r"\s+\b(?:hehe|Hehe|haha|HAHA|lol|LOL|lmao|LMAO|rofl|ROFL)\b[.!?]+\s*$", "", sentence)
  sentence = re.sub(r"\s{2,}", " ", sentence).strip()
  return sentence

def clean(text):
  text = text.astype(str).copy()

  # -------------------------
  # Unicode normalization
  # -------------------------
  text = text.str.translate(str.maketrans({
    '\u2018': "'",
    '\u2019': "'",
    '\u201a': "'",
    '\u201b': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\u201e': '"',
    '\u201f': '"',
    '\u00ab': '"',
    '\u00bb': '"',
    '\u2010': '-',
    '\u2011': '-',
    '\u2012': '-',
    '\u2013': '-',
    '\u2014': '-',
    '\u2212': '-',
    '\u2026': '...',
    '\u00a0': ' ',
  }))

  # Single-record text for the LLM.
  text = text.str.replace(r'[\r\n\t]+', ' ', regex=True)

  # Known malformed empty quote row.
  text = text.str.replace(
    r"^''' said Tom blankly\.$",
    '"" Tom said blankly.',
    regex=True
  )

  # Rows like: 'I love hot dogs,'' said Tom
  # Normalize opener before converting doubled apostrophes.
  text = text.str.replace(
    r"^'(?!')(?=.*'')",
    "''",
    regex=True
  )

  # Quote normalization.
  text = text.str.replace(r'``', '"', regex=True)
  text = text.str.replace(r"''", '"', regex=True)
  text = text.str.replace(r'^"\s*"', '""', regex=True)
  text = text.str.replace(r'"{3,}', '"', regex=True)

  # Convert single-quoted titles/dialogue to double quotes, but leave apostrophes in contractions alone.
  text = text.str.replace(r"(?<!\w)'([^'\n]{2,250})'(?!\w)", r'"\1"', regex=True)

  # Add missing space before an opening double quote: said"Hello" -> said "Hello".
  text = text.str.replace(r'(?<=[A-Za-z])"(?=[A-Za-z])', r' "', regex=True)

  # Remove web/social metadata.
  text = text.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
  text = text.str.replace(r'(?<!\w)@[A-Za-z0-9_]+\b:?\s*', '', regex=True)

  text = text.apply(_clean_hashtags_for_pun_recovery)
  text = text.apply(_remove_reaction_noise)

  # Punctuation and spacing.
  text = text.str.replace('……………..', '...', regex=False)
  text = text.str.replace(r'\s+([,.!?;:])', r'\1', regex=True)
  text = text.str.replace(r'([,.!?;:])(?=[A-Za-z])', r'\1 ', regex=True)
  text = text.str.replace(r",'", ", '", regex=True)

  # English dialogue punctuation: ", said -> ," said
  text = text.str.replace(r'",', ',"', regex=True)

  # Remove leading list/dialogue dash. Internal dialogue dashes remain.
  text = text.str.replace(r'^\s*[-–—]\s*', '', regex=True)

  # Fix import-tokenized compounds: high - tech -> high-tech.
  text = text.str.replace(r'(?<=\w)\s+-\s+(?=\w)', '-', regex=True)

  # Double hyphen as sentence break.
  text = text.str.replace(r'\s+--\s+', '. ', regex=True)

  # A : -> A:
  text = text.str.replace(r'\b([A-Z])\s+:\s*', r'\1: ', regex=True)

  # Pi artifacts.
  text = text.str.replace(r'\b3\.\s+(?=14\d)', '3.', regex=True)

  # Remove Shakespeare/location citation metadata that is not part of the pun.
  # Examples: (2.1.87-90), (4.2.23-28.), (1.4), (5.1.103-112)
  text = text.str.replace(r'\s*\(\d+(?:\.\d+)+(?:-\d+)?\.?\)', '', regex=True)

  # Normalize repeated punctuation for cleaner LLM parsing.
  # Keep ellipsis as exactly three dots; collapse !!!!/???? to one marker.
  text = text.str.replace(r'\.{4,}', '...', regex=True)
  text = text.str.replace(r'!{2,}', '!', regex=True)
  text = text.str.replace(r'\?{2,}', '?', regex=True)

  # Original useful special case.
  text = text.apply(_lower_old_puns)

  # Tom Swifty grammar repair before neutralization.
  text = text.apply(_repair_tom_before_neutralizing)

  # Possessive first, then standalone.
  text = text.str.replace(r"\bTom's\b", 'his', regex=True)
  text = text.str.replace(r"\bTom\b", 'he', regex=True)

  text = text.apply(_repair_after_neutralizing)

  # Keep full-row parenthetical jokes/comments because some contain the pun.

  # Late dataset-specific semantic/format repairs after quote/Tom normalization.
  text = text.apply(_fix_broken_contractions)
  text = text.apply(_manual_semantic_cleanups)
  text = text.str.replace(r'(?<!\.)\.\.(?!\.)', '.', regex=True)
  text = text.str.replace(r'(?<=[A-Za-z0-9?!.,])\s+"(?=[\s.,!?]|$)', '"', regex=True)
  text = text.apply(_normalize_single_quote_dialogue)
  text = text.apply(_manual_semantic_cleanups)
  text = text.str.replace(r'(?<!\.)\.\.(?!\.)', '.', regex=True)
  text = text.str.replace(r'(?<=[A-Za-z0-9?!.,])\s+"(?=[\s.,!?]|$)', '"', regex=True)

  # Final cleanup.
  text = (
    text.str.replace(r'\s{2,}', ' ', regex=True)
        .str.strip()
        .str.rstrip('\\')
        .str.strip()
  )

  return text



def _fix_broken_contractions(sentence):
  # Fix import/OCR apostrophe spacing without "correcting" pun spellings.
  # Examples: doesn ' t -> doesn't, you ' re -> you're.
  # This is intentionally about separated apostrophes only; it does not spell-correct words.
  sentence = re.sub(
    r"\b([A-Za-z]+)\s+'\s*(t|re|ve|ll|d|m|s)\b",
    lambda m: (
      f"{m.group(1)}'t" if m.group(2) == 't' and m.group(1).lower().endswith('n')
      else f"{m.group(1)}n't" if m.group(2) == 't'
      else f"{m.group(1)}'{m.group(2)}"
    ),
    sentence
  )
  return sentence


def _normalize_single_quote_dialogue(sentence):
  # Normalize remaining dialogue/title single quotes to double quotes for LLM consistency.
  # These rules avoid contractions/possessives by requiring a speech/title trigger.
  sentence = re.sub(
    r"\bsaying\s*'([^'\n]{2,120})'\s*\.?",
    lambda m: f'saying "{m.group(1).strip()}".',
    sentence,
    flags=re.IGNORECASE
  )
  sentence = re.sub(
    r"\bsaid\s+'([^'\n]{2,120})'\s*\.?",
    lambda m: f'said "{m.group(1).strip()}".',
    sentence,
    flags=re.IGNORECASE
  )
  sentence = re.sub(
    r"\bsaid\s+to\s+another\s+'([^'\n]{2,120})'\s*\.?",
    lambda m: f'said to another "{m.group(1).strip()}".',
    sentence,
    flags=re.IGNORECASE
  )
  sentence = re.sub(
    r"\breplied,\s*'([^'\n]{2,120})'\s*\.?",
    lambda m: f'replied, "{m.group(1).strip()}".',
    sentence,
    flags=re.IGNORECASE
  )
  # Directly adjacent title/dialogue cases: called'lightning' -> called "lightning".
  sentence = re.sub(
    r"\b(called|named|titled|reads|read|whispered)\s*'([^'\n]{2,120})'",
    lambda m: f'{m.group(1)} "{m.group(2).strip()}"',
    sentence,
    flags=re.IGNORECASE
  )
  return sentence


def _manual_semantic_cleanups(sentence):
  # Dataset-specific repairs that improve LLM readability without removing pun spellings.
  manual = {
    "A politician who had been an astronomer was always saying'no comet '.":
      'A politician who had been an astronomer was always saying "no comet".',
    "A politician who had been an astronomer was always saying'no comet '.":
      'A politician who had been an astronomer was always saying "no comet".',
    "One palm tree said to another 'let's have a date.'":
      "One palm tree said to another \"let's have a date.\"",
    "The sign on the nudist camp said, 'Clothed \"til May\".":
      "The sign on the nudist camp said, \"Clothed 'til May\".",
    '"It appears your vision is improving!" \'Really? "replied the patient."':
      '"It appears your vision is improving!" "Really?" replied the patient.',
    'Why do people say \'you can\'t trust atoms\'? Because they "make up everything"! Hehe!':
      'Why do people say "you can\'t trust atoms"? Because they "make up everything"!',
    "I don't really like coffee. It's just not my cup of tea'":
      "I don't really like coffee. It's just not my cup of tea.",
  }
  sentence = manual.get(sentence, sentence)

  # Strip a final unmatched single quote only when it is truly unmatched.
  if sentence.endswith("'") and sentence.count("'") % 2 == 1:
    sentence = sentence[:-1]

  return sentence



def validate_cleaned(df, text_col='text_clean'):
  problems = []

  for _, row in df.iterrows():
    id_en = str(row.get('id_en', ''))
    value = str(row.get(text_col, ''))

    if value == '':
      problems.append((id_en, 'empty_text', value))
    if value.endswith('\\'):
      problems.append((id_en, 'trailing_backslash', value))
    if '\\1' in value or '\\2' in value:
      problems.append((id_en, 'literal_backreference_left', value))
    if re.search(r'[“”‘’«»]', value):
      problems.append((id_en, 'curly_quote_left', value))
    if re.search(r'https?://|www\.', value):
      problems.append((id_en, 'url_left', value))
    if re.search(r'(?<!\w)@[A-Za-z0-9_]+\b', value):
      problems.append((id_en, 'handle_left', value))
    if re.search(r'\w\s+-\s+\w', value):
      problems.append((id_en, 'spaced_hyphen_left', value))
    if re.search(r'\(\d+(?:\.\d+)+(?:-\d+)?\.?\)', value):
      problems.append((id_en, 'citation_metadata_left', value))
    if re.search(r'\.{4,}|!{2,}|\?{2,}', value):
      problems.append((id_en, 'repeated_punctuation_left', value))
    if re.search(r'(?<!\.)\.\.(?!\.)', value):
      problems.append((id_en, 'double_period_left', value))
    if re.search(r'(?<=[A-Za-z0-9?!.,])\s+"(?=[\s.,!?]|$)', value):
      problems.append((id_en, 'space_before_closing_quote_left', value))
    if re.search(r"\b(?:called|said|says|asked|whispered|named|titled)'", value, re.IGNORECASE):
      problems.append((id_en, 'compressed_quote_left', value))
    if value.count('"') % 2 == 1:
      problems.append((id_en, 'unbalanced_double_quotes', value))
    if re.search(r'^[-–—]\s*', value):
      problems.append((id_en, 'leading_dash_left', value))
    if re.search(r"\b[A-Za-z]+\s+'\s+(?:t|re|ve|ll|d|m|s)\b", value):
      problems.append((id_en, 'broken_contraction_left', value))
    if re.search(r"(?:said|saying)\s*'[^']{2,100}'", value, re.IGNORECASE):
      problems.append((id_en, 'single_quote_dialogue_left', value))
    if re.search(r"[^\w]'\s*$", value):
      problems.append((id_en, 'trailing_stray_single_quote', value))
    if re.search(r"#[A-Za-z]", value):
      problems.append((id_en, 'raw_hashtag_left', value))
    if re.search(r"[\U0001F300-\U0001FAFF]", value):
      problems.append((id_en, 'emoji_left', value))
    if re.search(r"\b(?:hehe|Hehe|haha|HAHA|lol|LOL|lmao|LMAO|rofl|ROFL)\b[.!?]+\s*$", value):
      problems.append((id_en, 'reaction_noise_left', value))
    if re.search(r"\bTom\b|\bTom's\b", value):
      problems.append((id_en, 'tom_left', value))
    if re.search(rf"\b{SPEECH_VERBS}\s+he\s+", value, re.IGNORECASE):
      problems.append((id_en, 'bad_tom_swifty_grammar', value))

  audit_path = os.path.join(os.path.dirname(combined_en_path) or '.', 'cleaning_audit_problems.tsv')
  if problems:
    pd.DataFrame(problems, columns=['id_en', 'problem', 'text_clean']).to_csv(
      audit_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL
    )
    print(f'WARNING: validation found {len(problems)} potential issues.')
    print(f'Wrote audit: {audit_path}')
  else:
    # Remove stale audit from previous bad runs if it exists.
    if os.path.exists(audit_path):
      os.remove(audit_path)
    print('Validation passed: no structural cleaning problems found.')


def clean_en():
  translation_df = load(translation_path)
  translation_en_df = translation_df[~translation_df['id_en'].duplicated(keep='first')][['id_en', 'en']]
  translation_en_df['text_clean'] = clean(translation_en_df['en'])

  # Dataset-specific manual fixes after general cleaning.
  for id_en, fixed_text in MANUAL_TEXT_CLEAN.items():
    translation_en_df.loc[translation_en_df['id_en'].astype(str) == str(id_en), 'text_clean'] = fixed_text

  translation_en_df = translation_en_df.drop('en', axis=1)
  save(translation_en_df, cleaned_en_path)


def combine_en(translation_path=cleaned_en_path, save_path=combined_en_path):
  translation_df = load(translation_path)
  translation_df.drop(columns=['manual_location', 'manual_type', 'manual_alternative'], inplace=True, errors='ignore')
  location_manual_df = load(location_manual_path)

  translation_df['id_en'] = translation_df['id_en'].astype(str)
  location_manual_df['id_en'] = location_manual_df['id_en'].astype(str)

  location_df = pd.merge(translation_df, location_manual_df, left_on='id_en', right_on='id_en', how='left')

  save(location_df, save_path)
  validate_cleaned(location_df, text_col='text_clean')


def clean_fr():
  translation_fr_df = load(translation_path)
  translation_fr_df['text_clean'] = translation_fr_df['text_fr']
  save(translation_fr_df, cleaned_fr_path)


def combine_fr():
  translation_df = load(translation_path)
  location_input_df = load(location_fr_input_path)
  location_qrels_df = load(location_fr_qrels_path)

  combined_df = pd.merge(translation_df, location_input_df, left_on='text_fr', right_on='text', how='left')
  combined_df = pd.merge(combined_df, location_qrels_df, left_on='id', right_on='id', how='left')
  combined_df = combined_df[['id_en', 'text_fr', 'id', 'location']]
  combined_df = combined_df.groupby('id_en').agg(lambda x: list(x.dropna())).reset_index()

  save(combined_df, combined_fr_path)


if __name__ == "__main__":
  clean_en()
  combine_en()
  # clean_fr()
  # combine_fr()
