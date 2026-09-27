import streamlit as st
import streamlit.components.v1 as components

def render_3d_robot_avatar(user_name: str = ""):
    """
    Renders an interactive, glowing 3D Robot Avatar with dynamic introduction speech bubble.
    """
    greeting = f"Welcome back, {user_name}!" if user_name else "Welcome to MindSaathi App!"
    
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
        .robot-stage {{
            perspective: 800px;
            width: 200px;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 10px 0;
        }}
        .robot-body {{
            position: relative;
            width: 120px;
            height: 140px;
            transform-style: preserve-3d;
            animation: float3D 4s ease-in-out infinite alternate;
        }}
        /* Robot Head */
        .robot-head {{
            width: 90px;
            height: 75px;
            background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
            border: 2px solid #A855F7;
            border-radius: 20px;
            margin: 0 auto;
            position: relative;
            box-shadow: 0 0 30px rgba(168, 85, 247, 0.7), inset 0 0 15px rgba(236, 72, 153, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }}
        /* Antennas */
        .antenna {{
            position: absolute;
            top: -18px;
            left: 38px;
            width: 14px;
            height: 18px;
            border-left: 3px solid #EC4899;
        }}
        .antenna-ball {{
            position: absolute;
            top: -26px;
            left: 32px;
            width: 14px;
            height: 14px;
            background: #EC4899;
            border-radius: 50%;
            box-shadow: 0 0 15px #EC4899, 0 0 25px #F472B6;
            animation: pulseGlow 1.5s infinite alternate;
        }}
        /* Glowing Eyes */
        .eye {{
            width: 18px;
            height: 18px;
            background: #06B6D4;
            border-radius: 50%;
            box-shadow: 0 0 15px #06B6D4, 0 0 25px #22D3EE;
            animation: blinkEye 4s infinite;
        }}
        /* Torso */
        .robot-torso {{
            width: 105px;
            height: 65px;
            background: linear-gradient(135deg, #312E81 0%, #1E1B4B 100%);
            border: 2px solid #818CF8;
            border-radius: 16px;
            margin: 8px auto 0 auto;
            position: relative;
            box-shadow: 0 0 25px rgba(129, 140, 248, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .chest-core {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: radial-gradient(circle, #F472B6 0%, #9333EA 100%);
            box-shadow: 0 0 20px #EC4899;
            animation: corePulse 2s infinite alternate;
        }}
        /* Speech Bubble */
        .speech-bubble {{
            position: relative;
            background: rgba(30, 27, 75, 0.85);
            border: 1.5px solid rgba(168, 85, 247, 0.5);
            border-radius: 20px;
            padding: 16px 22px;
            max-width: 360px;
            text-align: center;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            margin-top: 10px;
        }}
        .speech-bubble::before {{
            content: '';
            position: absolute;
            top: -12px;
            left: 50%;
            transform: translateX(-50%);
            border-width: 0 12px 12px 12px;
            border-style: solid;
            border-color: transparent transparent rgba(30, 27, 75, 0.85) transparent;
        }}
        .intro-text {{
            font-size: 0.95rem;
            line-height: 1.5;
            color: #F3E8FF;
        }}
        .highlight {{
            color: #F472B6;
            font-weight: 700;
        }}

        @keyframes float3D {{
            0% {{ transform: translateY(0px) rotateY(-5deg); }}
            100% {{ transform: translateY(-12px) rotateY(5deg); }}
        }}
        @keyframes pulseGlow {{
            0% {{ transform: scale(0.9); opacity: 0.7; }}
            100% {{ transform: scale(1.2); opacity: 1; }}
        }}
        @keyframes corePulse {{
            0% {{ transform: scale(0.85); box-shadow: 0 0 10px #EC4899; }}
            100% {{ transform: scale(1.15); box-shadow: 0 0 25px #F472B6; }}
        }}
        @keyframes blinkEye {{
            0%, 48%, 52%, 100% {{ transform: scaleY(1); }}
            50% {{ transform: scaleY(0.1); }}
        }}
    </style>
    </head>
    <body>
        <div class="robot-stage">
            <div class="robot-body">
                <div class="antenna-ball"></div>
                <div class="antenna"></div>
                <div class="robot-head">
                    <div class="eye"></div>
                    <div class="eye"></div>
                </div>
                <div class="robot-torso">
                    <div class="chest-core"></div>
                </div>
            </div>
        </div>

        <div class="speech-bubble">
            <div class="intro-text">
                👋 <span class="highlight">{greeting}</span><br>
                I am your <b>MindSaathi AI Companion</b>. Feel free to ask me any question about exam stress, career anxiety, mental health, or personal wellbeing—I am here for you 24/7!
            </div>
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=360)
