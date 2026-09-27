import pytest
from modules.emotion_detector import detect_emotion, VALID_EMOTIONS

SAMPLE_MESSAGES = [
    ("I feel so sad and lonely in my hostel room", "sadness"),
    ("I am terrified of failing my upcoming semester exam", "fear"),
    ("I am so angry at my group project teammates", "anger"),
    ("I got selected for the campus placement job!", "joy"),
    ("What a surprising turn of events", "surprise")
]


def test_emotion_detector_output_structure():
    """Verify that emotion detector returns valid dict with label and score."""
    for text, expected in SAMPLE_MESSAGES:
        res = detect_emotion(text)
        assert isinstance(res, dict)
        assert "label" in res
        assert "score" in res
        assert res["label"] in VALID_EMOTIONS
        assert 0.0 <= res["score"] <= 1.0


def test_empty_input_handling():
    """Verify empty input returns neutral emotion."""
    res = detect_emotion("")
    assert res["label"] == "neutral"
    assert res["score"] == 1.0


if __name__ == "__main__":
    test_emotion_detector_output_structure()
    test_empty_input_handling()
    print("ALL EMOTION DETECTOR UNIT TESTS PASSED!")
