# PSS-4 Instrument Questions and Scoring Logic

PSS_QUESTIONS = [
    {
        "id": "pss1",
        "text": "In the last few minutes, how often have you felt that you were unable to control the important things in your life?",
        "reversed": False
    },
    {
        "id": "pss2",
        "text": "In the last few minutes, how often have you felt confident about your ability to handle your problems?",
        "reversed": True
    },
    {
        "id": "pss3",
        "text": "In the last few minutes, how often have you felt that things were going your way?",
        "reversed": True
    },
    {
        "id": "pss4",
        "text": "In the last few minutes, how often have you felt difficulties were piling up so high that you could not overcome them?",
        "reversed": False
    }
]

LIKERT_OPTIONS = {
    "Never": 0,
    "Almost Never": 1,
    "Sometimes": 2,
    "Fairly Often": 3,
    "Very Often": 4
}


def calculate_pss4_score(responses: dict) -> int:
    """
    Calculates total PSS-4 score from user responses dict.
    
    Args:
        responses (dict): e.g. {'pss1': 2, 'pss2': 3, 'pss3': 1, 'pss4': 2}
        
    Returns:
        int: Total PSS-4 score (range 0 to 16)
    """
    total = 0
    
    p1 = responses.get("pss1", 0)
    total += p1
    
    # Reverse coded item PSS2 (4 - score)
    p2 = responses.get("pss2", 0)
    total += (4 - p2)
    
    # Reverse coded item PSS3 (4 - score)
    p3 = responses.get("pss3", 0)
    total += (4 - p3)
    
    p4 = responses.get("pss4", 0)
    total += p4
    
    return total


def get_stress_severity_label(score: int) -> tuple:
    """
    Returns severity level text and color badge for a given PSS-4 score.
    """
    if score >= 14:
        return ("Severe Stress Level", "#E74C3C")
    elif score >= 10:
        return ("High Stress Level", "#E67E22")
    elif score >= 6:
        return ("Moderate Stress Level", "#F1C40F")
    else:
        return ("Low / Normal Stress Level", "#2ECC71")


EMOTION_MCQ_OPTIONS = [
    "Sadness",
    "Fear",
    "Anger",
    "Joy",
    "Surprise",
    "Disgust",
    "Neutral"
]

GENDER_OPTIONS = [
    "Female",
    "Male",
    "Non-binary",
    "Prefer not to say"
]

STREAM_OPTIONS = [
    "Engineering / Tech",
    "Medical / Healthcare",
    "Arts / Humanities",
    "Commerce / Business",
    "Law",
    "Basic Sciences",
    "Other"
]

CITY_TIER_OPTIONS = [
    "Metro City (Tier 1)",
    "Tier 2 City",
    "Tier 3 City",
    "Rural / Town"
]

WOULD_USE_AGAIN_OPTIONS = [
    "Yes, definitely",
    "Yes, maybe",
    "No"
]
