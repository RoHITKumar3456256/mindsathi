import re

# Comprehensive bilingual crisis keywords with flexible regex matching
ENGLISH_CRISIS_KEYWORDS = [
    r"\bsuicide\b",
    r"\bkill(ing)?\s+myself\b",
    r"\bend(ing)?\s+my\s+life\b",
    r"\bself[\s-]harm\b",
    r"\bwant\s+to\s+die\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bharm(ing)?\s+myself\b",
    r"\bcut(ting)?\s+myself\b",
    r"\boverdose\b",
    r"\bend\s+it\s+all\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bwish\s+i\s+(was|were)\s+dead\b"
]

HINGLISH_CRISIS_KEYWORDS = [
    r"\bkhatam\s+kar\b",
    r"\bjeena\s+nah?i\b",
    r"\bmar\s+jana\b",
    r"\bmarne\s+ka\b",
    r"\bkhud\s+ko\s+hurt\b",
    r"\bzindagi\s+se\s+thak\b",
    r"\bjaan\s+de\b",
    r"\bsuicide\b",
    r"\bmar\s+jaunga\b",
    r"\bapni\s+jaan\b"
]

ACADEMIC_CONTEXT_WORDS = [
    "exam", "test", "paper", "assignment", "syllabus", "marks", "grade",
    "subject", "jee", "neet", "upsc", "semester", "cgpa", "project", "presentation"
]

CRISIS_HELPLINES = {
    "iCall (TISS)": "9152987821 (Mon-Sat, 8:00 AM - 10:00 PM)",
    "Vandrevala Foundation": "1860-2662-345 / 9999-666-555 (24x7)",
    "NIMHANS Toll-Free": "080-46110007 (24x7)",
    "Tele-MANAS": "14416 / 1800-891-4416 (24x7)"
}

CRISIS_RESPONSE_TEXT = (
    "I hear how overwhelming things feel right now, and I want you to know that your life and safety matter deeply. "
    "Because I am an AI companion and cannot provide urgent crisis assistance, please connect immediately with trained human counselors who care and are ready to listen:\n\n"
    "📞 **iCall (TISS):** 9152987821\n"
    "📞 **Vandrevala Foundation:** 1860-2662-345 / 9999-666-555\n"
    "📞 **NIMHANS:** 080-46110007\n"
    "📞 **Tele-MANAS:** 14416\n\n"
    "Please reach out to one of these free, confidential helplines right away or speak to a trusted friend, family member, or hostel warden."
)


def check_crisis(message: str) -> dict:
    """
    Scans a user message for English and Hindi/Hinglish crisis triggers.
    Includes context check to prevent false positives (e.g. 'kill this exam').
    
    Returns:
        dict: {
            "is_crisis": bool,
            "response": str or None,
            "helplines": dict
        }
    """
    if not message or not isinstance(message, str):
        return {"is_crisis": False, "response": None, "helplines": CRISIS_HELPLINES}
    
    text_lower = message.lower().strip()
    
    # False positive check: e.g., "kill this exam" / "kill my test"
    is_academic_hyperbole = False
    if "kill" in text_lower:
        for word in ACADEMIC_CONTEXT_WORDS:
            if f"kill this {word}" in text_lower or f"kill my {word}" in text_lower or f"kill the {word}" in text_lower:
                is_academic_hyperbole = True
                break
    
    if is_academic_hyperbole:
        return {"is_crisis": False, "response": None, "helplines": CRISIS_HELPLINES}
    
    all_patterns = ENGLISH_CRISIS_KEYWORDS + HINGLISH_CRISIS_KEYWORDS
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            return {
                "is_crisis": True,
                "response": CRISIS_RESPONSE_TEXT,
                "helplines": CRISIS_HELPLINES
            }
            
    return {"is_crisis": False, "response": None, "helplines": CRISIS_HELPLINES}
