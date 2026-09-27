import os
import io
import json
import re
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)


def clean_text_for_speech(text: str) -> str:
    """Strips markdown syntax and special characters for natural voice synthesis."""
    if not text:
        return ""
    # Remove markdown links [text](url) -> text
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove markdown headers, bold, italics, code blocks
    t = re.sub(r'[*_#`~]', ' ', t)
    # Remove emojis
    t = re.sub(r'[^\w\s.,!?:;\'"-]', ' ', t)
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def transcribe_audio_groq(audio_file_or_bytes) -> str:
    """
    Transcribes audio using Groq's high-speed Whisper LPU engine (whisper-large-v3-turbo)
    in under 300ms with multilingual Hinglish/Hindi/English support.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ GROQ_API_KEY not found in .env. Please set GROQ_API_KEY to enable voice transcription.")
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        if hasattr(audio_file_or_bytes, "getvalue"):
            content = audio_file_or_bytes.getvalue()
            filename = getattr(audio_file_or_bytes, "name", "speech.wav") or "speech.wav"
        elif hasattr(audio_file_or_bytes, "read"):
            content = audio_file_or_bytes.read()
            filename = getattr(audio_file_or_bytes, "name", "speech.wav") or "speech.wav"
        else:
            content = audio_file_or_bytes
            filename = "speech.wav"
            
        if not content or len(content) < 100:
            return ""

        try:
            transcription = client.audio.transcriptions.create(
                file=(filename, content),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
            return str(transcription).strip()
        except Exception:
            # Fallback to default json response format
            res = client.audio.transcriptions.create(
                file=(filename, content),
                model="whisper-large-v3-turbo"
            )
            return str(getattr(res, "text", "")).strip()

    except Exception as e:
        print("Whisper transcription error:", e)
        return ""


def render_fluid_voice_visualizer(is_active: bool = False, text_to_speak: str = ""):
    """
    Renders the animated 3D fluid waveform canvas with user-controlled
    Play/Stop SpeechSynthesis audio playback.
    """
    cleaned_speech = clean_text_for_speech(text_to_speak)
    json_speech = json.dumps(cleaned_speech)
    display_caption = (cleaned_speech[:150] + "...") if len(cleaned_speech) > 150 else cleaned_speech
    json_caption = json.dumps("🔊 " + display_caption if display_caption else "Click record to speak or play MindSaathi's response.")
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            background: transparent;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: 'Inter', system-ui, sans-serif;
            color: #FFFFFF;
        }}
        .voice-header {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #00F5D4;
            margin-bottom: 4px;
            letter-spacing: 0.5px;
        }}
        .voice-sub {{
            font-size: 0.85rem;
            color: #94A3B8;
            margin-bottom: 10px;
        }}
        .canvas-container {{
            position: relative;
            width: 220px;
            height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        canvas {{
            border-radius: 50%;
            box-shadow: 0 0 45px rgba(0, 245, 212, 0.35), inset 0 0 25px rgba(155, 93, 229, 0.35);
        }}
        .controls-row {{
            display: flex;
            gap: 10px;
            margin-top: 14px;
        }}
        .voice-btn {{
            background: linear-gradient(135deg, rgba(0, 245, 212, 0.2), rgba(155, 93, 229, 0.2));
            color: #00F5D4;
            border: 1px solid rgba(0, 245, 212, 0.4);
            border-radius: 12px;
            padding: 8px 18px;
            font-size: 0.85rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .voice-btn:hover {{
            background: linear-gradient(135deg, #00F5D4, #9B5DE5);
            color: #0d0f1a;
            box-shadow: 0 4px 18px rgba(0, 245, 212, 0.4);
            transform: translateY(-2px);
        }}
        .stop-btn {{
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}
        .stop-btn:hover {{
            background: #EF4444;
            color: #FFFFFF;
            box-shadow: 0 4px 18px rgba(239, 68, 68, 0.4);
        }}
        .caption-box {{
            margin-top: 12px;
            font-size: 0.84rem;
            color: #E2D9F9;
            text-align: center;
            max-width: 360px;
            line-height: 1.4;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(155,89,245,0.25);
            border-radius: 12px;
            padding: 8px 14px;
        }}
    </style>
    </head>
    <body>
        <div class="voice-header">🎙️ AI Neural Voice Waveform</div>
        <div class="voice-sub">Groq Whisper (<300ms) & Multi-Agent Cognitive Voice</div>

        <div class="canvas-container">
            <canvas id="fluidCanvas" width="210" height="210"></canvas>
        </div>

        <div class="controls-row">
            <button class="voice-btn" id="btnPlay" onclick="speakAudio()">▶ Play Audio Reply</button>
            <button class="voice-btn stop-btn" id="btnStop" onclick="stopAudio()">⏹ Stop</button>
        </div>

        <div class="caption-box" id="captionText"></div>

        <script>
            const toSpeak = {json_speech};
            const caption = {json_caption};
            document.getElementById('captionText').innerText = caption;

            const canvas = document.getElementById('fluidCanvas');
            const ctx = canvas.getContext('2d');
            let step = 0;
            let isSpeaking = false;

            function drawFluidWave() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const cx = canvas.width / 2;
                const cy = canvas.height / 2;
                const radius = isSpeaking ? 62 : 56;
                const waveAmp = isSpeaking ? 16 : 8;

                ctx.beginPath();
                for (let i = 0; i <= 360; i += 4) {{
                    const angle = (i * Math.PI) / 180;
                    const offset = Math.sin((i * 4 + step) * Math.PI / 180) * waveAmp + Math.cos((i * 2 - step) * Math.PI / 180) * (waveAmp * 0.6);
                    const r = radius + offset;
                    const x = cx + r * Math.cos(angle);
                    const y = cy + r * Math.sin(angle);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.closePath();

                const gradient = ctx.createRadialGradient(cx, cy, 10, cx, cy, 95);
                gradient.addColorStop(0, '#0a0b14');
                gradient.addColorStop(0.3, '#1e1b4b');
                gradient.addColorStop(0.7, isSpeaking ? '#a855f7' : '#7c3aed');
                gradient.addColorStop(1, isSpeaking ? '#00f5d4' : '#06b6d4');

                ctx.fillStyle = gradient;
                ctx.fill();

                ctx.lineWidth = isSpeaking ? 3 : 2;
                ctx.strokeStyle = isSpeaking ? '#00F5D4' : '#38bdf8';
                ctx.stroke();

                step += isSpeaking ? 4.0 : 1.8;
                requestAnimationFrame(drawFluidWave);
            }}
            drawFluidWave();

            function speakAudio() {{
                if (!toSpeak || toSpeak.trim().length === 0) return;
                window.speechSynthesis.cancel();
                const utter = new SpeechSynthesisUtterance(toSpeak);
                utter.rate = 1.0;
                utter.pitch = 1.0;
                
                // Pick Indian or Hindi voice if available
                const voices = window.speechSynthesis.getVoices();
                const preferredVoice = voices.find(v => v.lang.includes('hi') || v.lang.includes('IN') || v.lang.includes('en-IN'));
                if (preferredVoice) utter.voice = preferredVoice;

                utter.onstart = () => {{ isSpeaking = true; }};
                utter.onend = () => {{ isSpeaking = false; }};
                utter.onerror = () => {{ isSpeaking = false; }};

                window.speechSynthesis.speak(utter);
            }}

            function stopAudio() {{
                window.speechSynthesis.cancel();
                isSpeaking = false;
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=380)


def render_audio_readout_button(text: str, msg_id: str):
    """Renders browser Web Speech TTS audio readout button with safe JSON encoding."""
    cleaned = clean_text_for_speech(text)
    json_text = json.dumps(cleaned)
    safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', msg_id)

    html_code = f"""
    <button id="btn_{safe_id}" onclick="speakText_{safe_id}()" style="
        background: rgba(0, 245, 212, 0.12);
        color: #00F5D4;
        border: 1px solid rgba(0, 245, 212, 0.35);
        padding: 4px 10px;
        border-radius: 10px;
        font-size: 0.76rem;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        margin-top: 6px;
        transition: all 0.2s;
    ">
        🔊 Listen
    </button>
    <script>
        function speakText_{safe_id}() {{
            window.speechSynthesis.cancel();
            const textToSpeak = {json_text};
            if (!textToSpeak) return;
            const utter = new SpeechSynthesisUtterance(textToSpeak);
            utter.rate = 1.0;
            const voices = window.speechSynthesis.getVoices();
            const preferred = voices.find(v => v.lang.includes('hi') || v.lang.includes('IN'));
            if (preferred) utter.voice = preferred;
            window.speechSynthesis.speak(utter);
        }}
    </script>
    """
    components.html(html_code, height=44)
