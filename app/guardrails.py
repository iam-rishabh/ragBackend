import re

INJECTION_PATTERNS = [
    r"system\s*prompt",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"reveal\s+your\s+(instructions|prompt|rules)",
    r"you\s+are\s+now\s+(a|an)\s",
    r"disregard\s+(all\s+)?(previous|prior)",
    r"print\s+(your\s+)?(system\s+)?prompt",
    r"what\s+(are|is)\s+your\s+(instructions|system\s+prompt|rules)",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _compiled)