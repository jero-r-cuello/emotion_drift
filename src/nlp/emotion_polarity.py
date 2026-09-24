"""Polarity (positive / negative / neutral-ambiguous) for every label we use.

Two things get mapped here:

* ``CONCEPT_POLARITY`` -- the 145 target emotion concepts the stimuli were
  generated from. Grouped by their parent in the \\citet{shaver_emotion_1987}
  hierarchy, so a concept inherits the polarity of its basic-level parent
  (love/joy -> positive, surprise -> ambiguous, anger/sadness/fear -> negative).
  Five concepts are deliberate exceptions to that inheritance; they are listed
  under OVERRIDES below, each with its reason.
* ``EKMAN_POLARITY`` / ``PLUTCHIK_POLARITY`` / ``GOEMOTIONS_POLARITY`` -- the
  judge's response labels. Ekman and GoEmotions are the maps already used in
  ``annotators_performance_analysis.py``; Plutchik is added here for parity.

Grouping the concepts by parent rather than listing them alphabetically is the
point: it makes the assignment checkable at a glance and makes it obvious which
entries are judgement calls rather than inheritance.
"""

POSITIVE = "positive"
NEGATIVE = "negative"
AMBIGUOUS = "neutral/ambiguous"

# --- Concepts, by Shaver parent -----------------------------------------------

_LOVE = [  # positive
    "adoration", "affection", "love", "fondness", "liking", "attraction",
    "caring", "tenderness", "compassion", "sentimentality",
    "desire", "lust", "passion", "infatuation",
]
_JOY = [  # positive
    "amusement", "bliss", "cheerfulness", "gaiety", "glee", "jolliness",
    "joviality", "joy", "delight", "enjoyment", "gladness", "happiness",
    "jubilation", "elation", "satisfaction", "ecstasy", "euphoria",
    "enthusiasm", "zeal", "zest", "excitement", "thrill", "exhilaration",
    "contentment", "pleasure", "pride", "triumph",
    "eagerness", "hope", "optimism", "enthrallment", "rapture", "relief",
]
_SURPRISE = [  # ambiguous: Shaver's own basic category is valence-free
    "amazement", "surprise", "astonishment",
]
_ANGER = [  # negative
    "aggravation", "irritation", "agitation", "annoyance", "grouchiness",
    "grumpiness", "exasperation", "frustration",
    "anger", "rage", "outrage", "fury", "wrath", "hostility", "ferocity",
    "bitterness", "hate", "loathing", "scorn", "spite", "vengefulness",
    "dislike", "resentment", "torment",
    "disgust", "revulsion", "contempt", "envy", "jealousy",
]
_SADNESS = [  # negative
    "agony", "suffering", "hurt", "anguish",
    "depression", "despair", "hopelessness", "gloom", "glumness", "sadness",
    "unhappiness", "grief", "sorrow", "woe", "misery", "melancholy",
    "dismay", "disappointment", "displeasure",
    "guilt", "shame", "regret", "remorse",
    "alienation", "isolation", "neglect", "loneliness", "rejection",
    "homesickness", "defeat", "dejection", "insecurity", "embarrassment",
    "humiliation", "insult",
]
_FEAR = [  # negative
    "alarm", "fear", "fright", "horror", "terror", "panic", "hysteria",
    "mortification",
    "anxiety", "nervousness", "tenseness", "uneasiness", "apprehension",
    "worry", "distress", "dread",
]

# Concepts with no Shaver entry, added ad-hoc from Plutchik / GoEmotions.
_ADDED = {
    "admiration": POSITIVE,
    "approval": POSITIVE,
    "gratitude": POSITIVE,
    "trust": POSITIVE,
    "disapproval": NEGATIVE,
    "anticipation": AMBIGUOUS,   # Plutchik: dreading and looking forward alike
    "curiosity": AMBIGUOUS,
    "confusion": AMBIGUOUS,
    "realization": AMBIGUOUS,
    "neutral": AMBIGUOUS,
}

# Departures from parent inheritance. Each is a case where the concept's own
# valence is not the valence of the branch Shaver files it under.
OVERRIDES = {
    # Shaver files these under sadness, but they name warmth directed at
    # someone else's misfortune, not the reader's own negative state.
    "sympathy": AMBIGUOUS,
    "pity": AMBIGUOUS,
    # Under love in Shaver, but longing is wanting what is absent: bittersweet.
    "longing": AMBIGUOUS,
    # Under fear/horror in Shaver, but a shock can be good or bad news.
    "shock": AMBIGUOUS,
    # Under lust/love in Shaver. In an affect-science context "arousal" reads
    # as the activation dimension, which is orthogonal to valence.
    "arousal": AMBIGUOUS,
}

CONCEPT_POLARITY = {}
for _group, _pol in ((_LOVE, POSITIVE), (_JOY, POSITIVE), (_SURPRISE, AMBIGUOUS),
                     (_ANGER, NEGATIVE), (_SADNESS, NEGATIVE), (_FEAR, NEGATIVE)):
    for _c in _group:
        CONCEPT_POLARITY[_c] = _pol
CONCEPT_POLARITY.update(_ADDED)
CONCEPT_POLARITY.update(OVERRIDES)

# --- Response labels ----------------------------------------------------------

EKMAN_POLARITY = {
    "disgust": NEGATIVE, "anger": NEGATIVE, "fear": NEGATIVE,
    "sadness": NEGATIVE, "enjoyment": POSITIVE,
    "surprise": AMBIGUOUS, "neutral": AMBIGUOUS,
}

PLUTCHIK_POLARITY = {
    "joy": POSITIVE, "trust": POSITIVE,
    "anger": NEGATIVE, "disgust": NEGATIVE, "fear": NEGATIVE,
    "sadness": NEGATIVE,
    "anticipation": AMBIGUOUS, "surprise": AMBIGUOUS, "neutral": AMBIGUOUS,
}

GOEMOTIONS_POLARITY = {
    "amusement": POSITIVE, "excitement": POSITIVE, "joy": POSITIVE,
    "love": POSITIVE, "desire": POSITIVE, "optimism": POSITIVE,
    "caring": POSITIVE, "pride": POSITIVE, "admiration": POSITIVE,
    "gratitude": POSITIVE, "relief": POSITIVE, "approval": POSITIVE,
    "realization": AMBIGUOUS, "surprise": AMBIGUOUS,
    "curiosity": AMBIGUOUS, "confusion": AMBIGUOUS, "neutral": AMBIGUOUS,
    "fear": NEGATIVE, "nervousness": NEGATIVE, "remorse": NEGATIVE,
    "embarrassment": NEGATIVE, "disappointment": NEGATIVE,
    "sadness": NEGATIVE, "grief": NEGATIVE, "disgust": NEGATIVE,
    "anger": NEGATIVE, "annoyance": NEGATIVE, "disapproval": NEGATIVE,
}

TAXONOMY_POLARITY = {
    "ekman_basic_emotions": EKMAN_POLARITY,
    "plutchik_wheel": PLUTCHIK_POLARITY,
    "go_emotions": GOEMOTIONS_POLARITY,
}

POLARITY_ORDER = [POSITIVE, AMBIGUOUS, NEGATIVE]

# Okabe-Ito blue / grey / vermillion. Red-green would collapse under the most
# common colour-vision deficiency, and this figure is read by polarity alone.
POLARITY_COLORS = {
    POSITIVE: "#0072B2",
    AMBIGUOUS: "#999999",
    NEGATIVE: "#D55E00",
}


def concept_polarity(name):
    """Polarity of a target emotion concept, or None if unmapped."""
    return CONCEPT_POLARITY.get(str(name).strip().lower())


def label_polarity(label, taxonomy):
    """Polarity of a judge label under one taxonomy, or None if unmapped."""
    return TAXONOMY_POLARITY[taxonomy].get(str(label).strip().lower())


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(CONCEPT_POLARITY)} concepts mapped")
    print(" ", Counter(CONCEPT_POLARITY.values()).most_common())
    print(f"{len(OVERRIDES)} overrides:", ", ".join(sorted(OVERRIDES)))
