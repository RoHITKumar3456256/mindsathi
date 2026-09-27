import re

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
VALID_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

_classifier_instance = None

def get_classifier():
    """
    Lazy loads the HuggingFace pipeline.
    """
    global _classifier_instance
    if _classifier_instance is None:
        try:
            from transformers import pipeline
            _classifier_instance = pipeline(
                "text-classification",
                model=MODEL_NAME,
                top_k=1,
                device=-1  # CPU inference
            )
        except Exception as e:
            # HuggingFace pipeline loading fallback
            _classifier_instance = "FALLBACK"
    return _classifier_instance


def _fallback_emotion_detector(text: str) -> dict:
    """
    Keyword-based fallback classifier when HuggingFace transformer model is unavailable offline.
    """
    text_lower = text.lower()
    
    sadness_kw = ["sad", "depressed", "lonely", "crying", "hopeless", "hurt", "broken", "upset", "dukhi", "akela", "rona"]
    fear_kw = ["scared", "fear", "anxious", "anxiety", "worried", "panic", "terrified", "darr", "nervous", "tension"]
    anger_kw = ["angry", "furious", "hate", "annoyed", "frustrated", "gussa", "irritated", "mad"]
    joy_kw = ["happy", "excited", "glad", "great", "awesome", "peaceful", "khush", "good", "relaxed"]
    surprise_kw = ["surprised", "shocked", "unexpected", "unbelievable", "heraan"]
    
    for kw in fear_kw:
        if kw in text_lower:
            return {"label": "fear", "score": 0.82}
    for kw in sadness_kw:
        if kw in text_lower:
            return {"label": "sadness", "score": 0.85}
    for kw in anger_kw:
        if kw in text_lower:
            return {"label": "anger", "score": 0.80}
    for kw in joy_kw:
        if kw in text_lower:
            return {"label": "joy", "score": 0.88}
    for kw in surprise_kw:
        if kw in text_lower:
            return {"label": "surprise", "score": 0.75}
            
    return {"label": "neutral", "score": 0.60}


def detect_emotion(text: str, custom_classifier=None) -> dict:
    """
    Classifies input text into one of 7 emotion categories.
    
    Rules:
      - Truncate text to max 512 chars/tokens.
      - If prediction confidence score < 0.45, label defaults to 'neutral'.
      - Guaranteed return of dict with 'label' and 'score'.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {"label": "neutral", "score": 1.0}
    
    truncated_text = text.strip()[:512]
    
    classifier = custom_classifier if custom_classifier else get_classifier()
    
    if classifier == "FALLBACK" or classifier is None:
        return _fallback_emotion_detector(truncated_text)
        
    try:
        results = classifier(truncated_text)
        # HuggingFace pipeline with top_k=1 returns [[{'label': 'sadness', 'score': 0.87}]]
        if isinstance(results, list) and len(results) > 0:
            item = results[0]
            if isinstance(item, list) and len(item) > 0:
                item = item[0]
            label = item.get("label", "neutral").lower()
            score = float(item.get("score", 0.0))
            
            # Confidence threshold override rule (< 0.45 -> neutral)
            if score < 0.45 or label not in VALID_EMOTIONS:
                label = "neutral"
                
            return {"label": label, "score": round(score, 4)}
    except Exception:
        pass
        
    return _fallback_emotion_detector(truncated_text)
