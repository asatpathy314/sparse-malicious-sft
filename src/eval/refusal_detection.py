REFUSAL_PHRASES = [
    "i cannot",
    "i can't",
    "i am unable",
    "i’m unable",
    "as an ai",
    "as a language model",
    "i cannot help with that",
    "i can't assist with that",
    "violates my safety",
    "goes against my guidelines",
    "i must refuse",
]


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in REFUSAL_PHRASES)
