import os
import time
from dotenv import load_dotenv

# Load environment variables — use explicit path so it works from any CWD
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "system_prompt.txt")

def load_system_prompt() -> str:
    """Reads system_prompt.txt from disk."""
    if os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "You are MindSaathi, a compassionate AI mental health companion for Indian college students."


def init_gemini():
    """Initializes and returns the Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
        
    # Prefer new google.genai SDK first
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return ("genai_new", client)
    except Exception:
        pass

    # Fallback to google.generativeai SDK
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return ("genai_old", genai)
    except Exception:
        pass
        
    return None


def get_response(user_message: str, emotion: str, history: list = None, user_memory: str = "", model_client = None) -> str:
    """
    Generates dynamic response via MindSaathi Multi-Agent Neural Orchestrator:
    - Tier 1: Ling 3.0 Flash Sante (Health MoE 124B)
    - Tier 2: Google Gemini 2.5 Flash (1M Context)
    - Tier 3: Groq LPU Ultra-Fast (<500ms zero-downtime)
    - Tier 4: Thinking Machines Inkling
    - Tier 5: Clinical CBT core
    """
    from modules.orchestrator import route_and_respond
    result = route_and_respond(
        user_message=user_message,
        emotion=emotion,
        history=history,
        user_memory=user_memory
    )
    return result.get("response", "")



def _get_fallback_cbt_response(user_message: str, emotion: str) -> str:
    """Fallback empathetic CBT response if network is offline."""
    msg_lower = user_message.lower()
    
    if "exam" in msg_lower or "test" in msg_lower or "marks" in msg_lower or "paper" in msg_lower:
        return (
            "I hear how much pressure you are feeling about your exams right now. "
            "Academic expectations in India can feel incredibly heavy, but your worth is never defined by a single test score. "
            "Take a deep breath and break your preparation down into small, achievable steps. "
            "How can we break down your study goals for today into just one manageable task?"
        )
    elif "family" in msg_lower or "parent" in msg_lower or "ghar" in msg_lower:
        return (
            "Family expectations and the fear of letting loved ones down can weigh so heavily on your mind. "
            "It takes immense strength to balance your own aspirations with family desires. "
            "Remember that your personal wellbeing matters just as much as fulfilling external hopes. "
            "What is one thought you wish you could share with your family about how you feel?"
        )
    elif "lonely" in msg_lower or "hostel" in msg_lower or "friend" in msg_lower:
        return (
            "Hostel life and feeling disconnected from people around you can feel really isolating. "
            "Even in crowded campuses, many students carry this exact same quiet loneliness. "
            "Be gentle with yourself as you navigate building meaningful connections at your own pace. "
            "What is one small comfort or activity that makes you feel a bit more grounded today?"
        )
    else:
        return (
            f"Thank you for sharing that with me. It sounds like you are carrying a lot of emotion right now, and that is completely valid. "
            "When thoughts feel overwhelming, taking a slow, deep breath can help bring a moment of clarity. "
            "What feels like the heaviest part of this situation for you right now?"
        )
