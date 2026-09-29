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
    Renders the Pinterest-style 3D Iridescent Holographic Liquid Sphere
    with fluid reflections, voice analysis state, and audio readout playback.
    """
    cleaned_speech = clean_text_for_speech(text_to_speak)
    json_speech = json.dumps(cleaned_speech)
    display_caption = (cleaned_speech[:140] + "...") if len(cleaned_speech) > 140 else cleaned_speech
    json_caption = json.dumps(display_caption if display_caption else "What is on your mind? Share whatever you are feeling without hesitation...")
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
        
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 8px;
            background: transparent;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            color: #1E1B2E;
        }}
        .voice-card-container {{
            background: radial-gradient(circle at 50% 20%, rgba(245, 232, 255, 0.95), rgba(255, 255, 255, 0.98));
            border: 1.5px solid rgba(226, 218, 248, 0.85);
            border-radius: 32px;
            padding: 24px 20px;
            width: 100%;
            max-width: 380px;
            box-shadow: 0 16px 40px -8px rgba(139, 92, 246, 0.12), 0 4px 16px -2px rgba(139, 92, 246, 0.05);
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            overflow: hidden;
        }}
        .voice-top-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            margin-bottom: 8px;
            padding: 0 6px;
        }}
        .voice-title-pill {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #1E1B2E;
            letter-spacing: -0.01em;
        }}
        .voice-live-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.25);
            color: #7C3AED;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 600;
        }}
        .live-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #8B5CF6;
            box-shadow: 0 0 8px #8B5CF6;
            animation: pulseDot 2s infinite ease-in-out;
        }}
        @keyframes pulseDot {{
            0%, 100% {{ transform: scale(1); opacity: 0.8; }}
            50% {{ transform: scale(1.3); opacity: 1; }}
        }}
        .orb-stage {{
            position: relative;
            width: 220px;
            height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 12px 0;
        }}
        canvas {{
            border-radius: 50%;
            filter: drop-shadow(0 15px 35px rgba(139, 92, 246, 0.28));
        }}
        .voice-prompt-quote {{
            font-size: 0.88rem;
            color: #4B5563;
            text-align: center;
            line-height: 1.45;
            margin: 10px 0 16px 0;
            padding: 0 12px;
            font-weight: 500;
            min-height: 42px;
        }}
        .sound-wave-pill {{
            background: #1E1B2E;
            color: #FFFFFF;
            border-radius: 9999px;
            padding: 8px 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 8px 24px rgba(30, 27, 46, 0.25);
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }}
        .sound-wave-pill:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(30, 27, 46, 0.35);
        }}
        .wave-bars {{
            display: flex;
            align-items: center;
            gap: 2.5px;
            height: 18px;
        }}
        .wave-bar {{
            width: 2.5px;
            background: linear-gradient(180deg, #A855F7, #EC4899);
            border-radius: 3px;
            animation: waveBounce 1.2s infinite ease-in-out alternate;
        }}
        .wave-bar:nth-child(2) {{ height: 14px; animation-delay: 0.1s; }}
        .wave-bar:nth-child(3) {{ height: 8px; animation-delay: 0.2s; }}
        .wave-bar:nth-child(4) {{ height: 16px; animation-delay: 0.3s; }}
        .wave-bar:nth-child(5) {{ height: 11px; animation-delay: 0.15s; }}
        .wave-bar:nth-child(6) {{ height: 18px; animation-delay: 0.4s; }}
        .wave-bar:nth-child(7) {{ height: 7px; animation-delay: 0.25s; }}
        .wave-bar:nth-child(8) {{ height: 13px; animation-delay: 0.35s; }}
        @keyframes waveBounce {{
            0% {{ transform: scaleY(0.4); }}
            100% {{ transform: scaleY(1); }}
        }}
        .wave-timer {{
            font-size: 0.78rem;
            color: #D1D5DB;
            font-weight: 600;
            letter-spacing: 0.04em;
        }}
        .play-btn-circle {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #8B5CF6;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.68rem;
            border: none;
            cursor: pointer;
        }}
        .controls-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
            justify-content: center;
        }}
        .circle-btn {{
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: #FFFFFF;
            border: 1.5px solid rgba(226, 218, 248, 0.9);
            color: #4B5563;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.08);
            transition: all 0.2s ease;
        }}
        .circle-btn:hover {{
            background: #F3E8FF;
            color: #7C3AED;
            transform: scale(1.05);
        }}
        .mic-btn-main {{
            width: 58px;
            height: 58px;
            border-radius: 50%;
            background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
            border: none;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.35rem;
            cursor: pointer;
            box-shadow: 0 8px 24px rgba(124, 58, 237, 0.35);
            transition: all 0.22s ease;
        }}
        .mic-btn-main:hover {{
            background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
            box-shadow: 0 12px 30px rgba(124, 58, 237, 0.5);
            transform: scale(1.08);
        }}
    </style>
    </head>
    <body>
        <div class="voice-card-container">
            <div class="voice-top-header">
                <span class="voice-title-pill">Voice Analysis</span>
                <span class="voice-live-badge"><span class="live-dot"></span> Listening...</span>
            </div>

            <div class="orb-stage">
                <canvas id="orbCanvas" width="200" height="200"></canvas>
            </div>

            <div class="voice-prompt-quote" id="captionText"></div>

            <div class="sound-wave-pill" onclick="toggleAudio()">
                <div class="wave-bars">
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                </div>
                <span class="wave-timer" id="audioStatusText">00:05</span>
                <button class="play-btn-circle" id="playIcon">▶</button>
            </div>

            <div class="controls-row">
                <button class="circle-btn" onclick="stopAudio()" title="Pause">⏸</button>
                <button class="mic-btn-main" onclick="toggleAudio()" title="Speak / Listen">🎙️</button>
                <button class="circle-btn" onclick="stopAudio()" title="Clear">✕</button>
            </div>
        </div>

        <script>
            const toSpeak = {json_speech};
            const caption = {json_caption};
            document.getElementById('captionText').innerText = caption;

            const canvas = document.getElementById('orbCanvas');
            const ctx = canvas.getContext('2d');
            let t = 0;
            let isSpeaking = false;

            function drawHolographicOrb() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const cx = canvas.width / 2;
                const cy = canvas.height / 2;
                const r = 76;
                const waveAmp = isSpeaking ? 6 : 2.5;

                // 1. Soft Outer Atmosphere Glow
                const glow = ctx.createRadialGradient(cx, cy, r * 0.7, cx, cy, r * 1.25);
                glow.addColorStop(0, 'rgba(168, 85, 247, 0.35)');
                glow.addColorStop(0.6, 'rgba(236, 72, 153, 0.18)');
                glow.addColorStop(1, 'rgba(255, 255, 255, 0)');
                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(cx, cy, r * 1.25, 0, Math.PI * 2);
                ctx.fill();

                // 2. Dynamic Liquid Surface Path
                ctx.beginPath();
                for (let i = 0; i <= 360; i += 3) {{
                    const rad = (i * Math.PI) / 180;
                    const offset = Math.sin(rad * 4 + t * 0.05) * waveAmp 
                                 + Math.cos(rad * 6 - t * 0.04) * (waveAmp * 0.6)
                                 + Math.sin(rad * 2 + t * 0.02) * (waveAmp * 0.4);
                    const currentR = r + offset;
                    const x = cx + currentR * Math.cos(rad);
                    const y = cy + currentR * Math.sin(rad);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.closePath();

                // 3. Iridescent Swirling Sphere Shading
                const sphereGrad = ctx.createRadialGradient(cx - 24, cy - 24, 10, cx, cy, r);
                sphereGrad.addColorStop(0, '#FFFFFF'); // Specular highlight
                sphereGrad.addColorStop(0.18, '#FDE8E9'); // Gentle rose pearl
                sphereGrad.addColorStop(0.42, '#C4B5FD'); // Soft lilac
                sphereGrad.addColorStop(0.68, '#818CF8'); // Ethereal azure violet
                sphereGrad.addColorStop(0.92, '#C084FC'); // Radiant purple
                sphereGrad.addColorStop(1.0, '#7C3AED'); // Deep violet rim
                ctx.fillStyle = sphereGrad;
                ctx.fill();

                // 4. Liquid Chromatic Shimmer Rings
                ctx.save();
                ctx.clip(); // Clip inside sphere
                
                // Chrome shimmer band 1
                ctx.beginPath();
                const bandY1 = cy + Math.sin(t * 0.03) * 25;
                ctx.ellipse(cx, bandY1, r * 0.95, 26, Math.PI / 6, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
                ctx.lineWidth = 14;
                ctx.filter = 'blur(6px)';
                ctx.stroke();

                // Chrome shimmer band 2 (cyan sheen)
                ctx.beginPath();
                const bandY2 = cy - Math.cos(t * 0.025) * 22;
                ctx.ellipse(cx, bandY2, r * 0.85, 20, -Math.PI / 4, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(125, 211, 252, 0.4)';
                ctx.lineWidth = 10;
                ctx.stroke();

                // Chrome shimmer band 3 (magenta sheen)
                ctx.beginPath();
                ctx.ellipse(cx - 10, cy + 10, r * 0.7, 16, Math.PI / 3, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(244, 114, 182, 0.35)';
                ctx.lineWidth = 8;
                ctx.stroke();

                ctx.restore();

                // 5. Crisp Specular Pearl Highlight
                ctx.beginPath();
                ctx.arc(cx - 30, cy - 30, 16, 0, Math.PI * 2);
                const specGrad = ctx.createRadialGradient(cx - 30, cy - 30, 1, cx - 30, cy - 30, 16);
                specGrad.addColorStop(0, 'rgba(255, 255, 255, 0.85)');
                specGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
                ctx.fillStyle = specGrad;
                ctx.fill();

                t += isSpeaking ? 1.8 : 0.8;
                requestAnimationFrame(drawHolographicOrb);
            }}
            drawHolographicOrb();

            function toggleAudio() {{
                if (isSpeaking) {{
                    stopAudio();
                }} else {{
                    speakAudio();
                }}
            }}

            function speakAudio() {{
                if (!toSpeak || toSpeak.trim().length === 0) return;
                window.speechSynthesis.cancel();
                const utter = new SpeechSynthesisUtterance(toSpeak);
                utter.rate = 1.0;
                utter.pitch = 1.0;
                
                const voices = window.speechSynthesis.getVoices();
                const preferredVoice = voices.find(v => v.lang.includes('hi') || v.lang.includes('IN') || v.lang.includes('en-IN'));
                if (preferredVoice) utter.voice = preferredVoice;

                utter.onstart = () => {{
                    isSpeaking = true;
                    document.getElementById('playIcon').innerText = '⏹';
                    document.getElementById('audioStatusText').innerText = 'Speaking...';
                }};
                utter.onend = () => {{
                    isSpeaking = false;
                    document.getElementById('playIcon').innerText = '▶';
                    document.getElementById('audioStatusText').innerText = '00:05';
                }};
                utter.onerror = () => {{
                    isSpeaking = false;
                    document.getElementById('playIcon').innerText = '▶';
                    document.getElementById('audioStatusText').innerText = '00:05';
                }};

                window.speechSynthesis.speak(utter);
            }}

            function stopAudio() {{
                window.speechSynthesis.cancel();
                isSpeaking = false;
                document.getElementById('playIcon').innerText = '▶';
                document.getElementById('audioStatusText').innerText = '00:05';
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=520)


def render_audio_readout_button(text: str, msg_id: str):
    """Renders the Pinterest-style dark sleek audio player pill with animated waveform and play button."""
    cleaned = clean_text_for_speech(text)
    json_text = json.dumps(cleaned)
    safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', msg_id)

    html_code = f"""
    <div style="margin-top: 8px;">
        <button id="btn_{safe_id}" onclick="togglePlay_{safe_id}()" style="
            background: #1E1B2E;
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 6px 18px rgba(30, 27, 46, 0.22);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        ">
            <span id="waves_{safe_id}" style="color: #A855F7; letter-spacing: 1px; font-size: 0.85rem;">ılılılllı</span>
            <span id="label_{safe_id}">00:05</span>
            <span id="icon_{safe_id}" style="
                background: #8B5CF6;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 0.62rem;
                margin-left: 2px;
            ">▶</span>
        </button>
    </div>
    <script>
        let isPlaying_{safe_id} = false;

        function togglePlay_{safe_id}() {{
            if (isPlaying_{safe_id}) {{
                window.speechSynthesis.cancel();
                isPlaying_{safe_id} = false;
                document.getElementById('label_{safe_id}').innerText = '00:05';
                document.getElementById('icon_{safe_id}').innerText = '▶';
            }} else {{
                window.speechSynthesis.cancel();
                const textToSpeak = {json_text};
                if (!textToSpeak) return;
                const utter = new SpeechSynthesisUtterance(textToSpeak);
                utter.rate = 1.0;
                const voices = window.speechSynthesis.getVoices();
                const preferred = voices.find(v => v.lang.includes('hi') || v.lang.includes('IN'));
                if (preferred) utter.voice = preferred;

                utter.onstart = () => {{
                    isPlaying_{safe_id} = true;
                    document.getElementById('label_{safe_id}').innerText = 'Playing';
                    document.getElementById('icon_{safe_id}').innerText = '⏹';
                }};
                utter.onend = () => {{
                    isPlaying_{safe_id} = false;
                    document.getElementById('label_{safe_id}').innerText = '00:05';
                    document.getElementById('icon_{safe_id}').innerText = '▶';
                }};
                utter.onerror = () => {{
                    isPlaying_{safe_id} = false;
                    document.getElementById('label_{safe_id}').innerText = '00:05';
                    document.getElementById('icon_{safe_id}').innerText = '▶';
                }};

                window.speechSynthesis.speak(utter);
            }}
        }}
    </script>
    """
    components.html(html_code, height=54)

