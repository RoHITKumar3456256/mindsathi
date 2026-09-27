import pytest
from modules.crisis_handler import check_crisis

CRISIS_PHRASES = [
    "I want to suicide",
    "I feel like killing myself tonight",
    "I want to end my life",
    "I am thinking of self harm",
    "I want to die, nothing matters anymore",
    "There is no reason to live",
    "I want to cut myself",
    "I might take an overdose",
    "Mujhe khatam kar lene ka man kar raha hai",
    "Ab aur jeena nahi hai mujhe",
    "Main mar jana chahta hu",
    "Zindagi se thak gaya hu, sab khatam kar dunga",
    "Aaj raat marne ka man kar raha hai",
    "Apni jaan de dunga",
    "Suicide kar lunga main"
]

NON_CRISIS_PHRASES = [
    "I want to kill this exam tomorrow!",
    "I need to slay this test",
    "I feel so tired after class",
    "Exam ka tension ho raha hai",
    "Hostel food is terrible"
]


def test_crisis_triggers():
    """Verify that all crisis phrases trigger is_crisis=True (0% false negatives)."""
    for phrase in CRISIS_PHRASES:
        res = check_crisis(phrase)
        assert res["is_crisis"] is True, f"Failed to detect crisis phrase: '{phrase}'"
        assert res["response"] is not None
        assert "9152987821" in res["response"]


def test_non_crisis_phrases():
    """Verify that academic hyperbole and normal stress do NOT trigger false crisis alerts."""
    for phrase in NON_CRISIS_PHRASES:
        res = check_crisis(phrase)
        assert res["is_crisis"] is False, f"False positive crisis triggered on: '{phrase}'"


if __name__ == "__main__":
    test_crisis_triggers()
    test_non_crisis_phrases()
    print("ALL CRISIS HANDLER UNIT TESTS PASSED (100% Detection, 0 False Negatives)!")
