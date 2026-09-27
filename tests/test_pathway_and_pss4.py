import pytest
from modules.survey import calculate_pss4_score, get_stress_severity_label
from modules.pathway_generator import detect_pathway_intent, render_visual_pathway_svg, PATHWAY_TEMPLATES
from modules.voice_ui import clean_text_for_speech


def test_pss4_scoring_and_reverse_coding():
    """Verify Cohen 1983 PSS-4 scoring with reverse items (pss2 and pss3)."""
    # All 0s: pss1=0, pss2=0(rev:4), pss3=0(rev:4), pss4=0 -> total = 8
    score_all_zero = calculate_pss4_score({"pss1": 0, "pss2": 0, "pss3": 0, "pss4": 0})
    assert score_all_zero == 8

    # All max stress: pss1=4, pss2=0(rev:4), pss3=0(rev:4), pss4=4 -> total = 16
    score_max_stress = calculate_pss4_score({"pss1": 4, "pss2": 0, "pss3": 0, "pss4": 4})
    assert score_max_stress == 16

    # All min stress: pss1=0, pss2=4(rev:0), pss3=4(rev:0), pss4=0 -> total = 0
    score_min_stress = calculate_pss4_score({"pss1": 0, "pss2": 4, "pss3": 4, "pss4": 0})
    assert score_min_stress == 0


def test_stress_severity_labels():
    """Verify clinical stress labels and color badges."""
    assert get_stress_severity_label(15)[0] == "Severe Stress Level"
    assert get_stress_severity_label(11)[0] == "High Stress Level"
    assert get_stress_severity_label(7)[0] == "Moderate Stress Level"
    assert get_stress_severity_label(3)[0] == "Low / Normal Stress Level"


def test_pathway_intent_detection():
    """Verify pathway triggers for various student and mental health queries."""
    assert detect_pathway_intent("mujhe exam anxiety ka roadmap dikhao") == "exam"
    assert detect_pathway_intent("show me career placement steps") == "career"
    assert detect_pathway_intent("neend nahi aa rahi sleep routine path batao") == "sleep"
    assert detect_pathway_intent("procrastination todne ka path") == "focus"
    assert detect_pathway_intent("breakup ke baad recovery roadmap") == "breakup"
    assert detect_pathway_intent("burnout ho gaya recharge pathway") == "burnout"
    assert detect_pathway_intent("anxiety ground karne ka rasta") == "anxiety"


def test_render_visual_pathway_svg():
    """Verify SVG roadmap generation produces valid SVG markup."""
    for p_type in PATHWAY_TEMPLATES.keys():
        svg = render_visual_pathway_svg(p_type)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "xmlns=\"http://www.w3.org/2000/svg\"" in svg
        assert "linearGradient id=\"pathGrad\"" in svg


def test_clean_text_for_speech():
    """Verify markdown symbols and emojis are stripped for natural voice synthesis."""
    raw = "**Namaste!** Please visit [MindSaathi](https://mindsathi.app) `now` #cbt 🌟"
    cleaned = clean_text_for_speech(raw)
    assert "*" not in cleaned
    assert "#" not in cleaned
    assert "`" not in cleaned
    assert "Namaste" in cleaned
    assert "MindSaathi" in cleaned
