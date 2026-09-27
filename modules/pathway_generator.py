"""
MindSaathi Visual Pathway & Roadmap Generator
Generates visual SVG step-by-step guidance diagrams and roadmaps
for stress relief, exam anxiety, career planning, emotional recovery, and mindfulness.
"""

import html
from typing import List, Dict, Optional, Tuple


def _xml_escape(text: str) -> str:
    """Escapes special XML/SVG characters so the XML parser never crashes."""
    return html.escape(str(text), quote=True)


PATHWAY_TEMPLATES = {
    "exam": {
        "title": "📚 5-Step Exam Anxiety Recovery Pathway",
        "description": "Evidence-based CBT protocol to calm exam nerves and optimize focus",
        "steps": [
            ("Step 1: Grounding and Reset", "4-7-8 Breathing to calm the sympathetic nervous system.", "🧘"),
            ("Step 2: Micro-Chunking", "Break syllabus into 25-minute Pomodoro sprints.", "⏱️"),
            ("Step 3: Active Recall", "Test memory with flashcards instead of passive reading.", "🧠"),
            ("Step 4: Energy Management", "Hydrate, light walking, and 7 hours protected sleep.", "⚡"),
            ("Step 5: Exam Day Anchoring", "Positive self-talk: My worth is not defined by one score.", "🌟")
        ]
    },
    "anxiety": {
        "title": "🌿 5-4-3-2-1 Sensory Grounding Pathway",
        "description": "Fast neuro-somatic reset to dissolve acute panic and anxiety spirals",
        "steps": [
            ("Step 1: 5 Things You See", "Look around and name 5 specific objects in your room.", "👀"),
            ("Step 2: 4 Things You Feel", "Touch your desk, feet on floor, cool air on skin.", "✋"),
            ("Step 3: 3 Things You Hear", "Listen for distant traffic, fan hum, or birds outside.", "👂"),
            ("Step 4: 2 Things You Smell", "Notice coffee aroma, fresh pencil, or your clothing.", "👃"),
            ("Step 5: 1 Thing You Taste", "Sip cold water or focus on the clean breath in your mouth.", "👅")
        ]
    },
    "career": {
        "title": "🚀 Placement and Career Clarity Roadmap",
        "description": "Structured steps to overcome career paralysis and job search anxiety",
        "steps": [
            ("Step 1: Core Strength Audit", "Identify 3 skills you genuinely enjoy and are good at.", "🎯"),
            ("Step 2: Resume and Proof of Work", "Build 2 clean portfolio projects that showcase impact.", "💻"),
            ("Step 3: Network and Outreach", "Connect with 5 alumni on LinkedIn for coffee chats.", "🤝"),
            ("Step 4: Mock Interview Reps", "Practice technical and behavioral questions out loud.", "🗣️"),
            ("Step 5: Mindset Calibration", "Rejection is redirection — every interview is valuable data.", "🛡️")
        ]
    },
    "sleep": {
        "title": "🌙 Sleep and De-Stress Night Routine Pathway",
        "description": "Sleep hygiene protocol to turn off a racing mind at night",
        "steps": [
            ("Step 1: Digital Sunset (T - 60m)", "Turn off screens or switch to ultra-warm night shift.", "📵"),
            ("Step 2: Brain Dump Journaling", "Write down all racing thoughts on paper to clear RAM.", "📝"),
            ("Step 3: Temperature Drop", "Take a warm shower or wash face to trigger melatonin.", "☕"),
            ("Step 4: Progressive Relaxation", "Tense and release toes to forehead slowly in bed.", "🛏️"),
            ("Step 5: Deep Delta Sleep", "Allow your brain to repair and synthesize memories.", "✨")
        ]
    },
    "focus": {
        "title": "🎯 Anti-Procrastination and Focus Sprint Pathway",
        "description": "Beat dopamine overload and executive dysfunction",
        "steps": [
            ("Step 1: 2-Minute Rule", "Just start for 120 seconds — starting breaks inertia.", "⚡"),
            ("Step 2: Single-Task Shield", "Close all browser tabs except the current task.", "🛡️"),
            ("Step 3: 25/5 Pomodoro Cycle", "25 minutes intense focus, then 5 minutes screen-free break.", "⏱️"),
            ("Step 4: Dopamine Reward", "Reward your brain with green tea or music after 2 cycles.", "☕"),
            ("Step 5: Daily Victory Log", "Write down 3 things you finished today.", "🏆")
        ]
    },
    "breakup": {
        "title": "💔 Emotional Healing and Heartbreak Pathway",
        "description": "Gentle, self-compassionate recovery after a relationship ends",
        "steps": [
            ("Step 1: Safe Expression", "Let yourself cry and feel without self-judgment.", "🌧️"),
            ("Step 2: Digital Detox", "Mute/archive chats and photos to give your heart space.", "📱"),
            ("Step 3: Reconnecting with Self", "Spend time on hobbies you set aside during the relationship.", "🎨"),
            ("Step 4: Circle of Support", "Talk to 1-2 trusted friends or your MindSaathi companion.", "🫂"),
            ("Step 5: Future Re-imagining", "You are whole on your own. New beginnings await.", "🌅")
        ]
    },
    "burnout": {
        "title": "🔋 Academic Burnout and Nervous System Recharge",
        "description": "Rebuild mental stamina after prolonged semester exhaustion",
        "steps": [
            ("Step 1: Radical Rest Day", "One full day with zero guilt and zero academic study.", "🛋️"),
            ("Step 2: Boundary Setting", "Learn to say 'No' to non-essential commitments.", "🛑"),
            ("Step 3: Nature Immersion", "20 minutes morning sunlight and bare-foot grounding.", "🌳"),
            ("Step 4: Micro-Pacing", "Cut study hours by 30% and focus on high-yield tasks.", "⚖️"),
            ("Step 5: Joy Spark", "Do one silly, fun activity just for the joy of it.", "🎈")
        ]
    }
}


def detect_pathway_intent(user_message: str) -> Optional[str]:
    """Detects if user is asking for a pathway, roadmap, or step-by-step guidance."""
    msg = user_message.lower()
    triggers = [
        "path", "roadmap", "rasta", "kadam", "steps", "plan", "road map",
        "guide", "flowchart", "tarika", "kaise karu", "how to prepare",
        "grounding", "batao", "dikhao", "solution", "sprint"
    ]
    if not any(t in msg for t in triggers):
        return None
    
    if any(k in msg for k in ["exam", "test", "padhai", "study", "marks", "paper"]):
        return "exam"
    elif any(k in msg for k in ["career", "placement", "job", "internship", "interview", "resume"]):
        return "career"
    elif any(k in msg for k in ["sleep", "nind", "insomnia", "night", "so nahi", "neend"]):
        return "sleep"
    elif any(k in msg for k in ["focus", "procrastin", "taldol", "dhyan", "concentrat"]):
        return "focus"
    elif any(k in msg for k in ["breakup", "heartbreak", "relationship", "dhokha", "judai"]):
        return "breakup"
    elif any(k in msg for k in ["burnout", "exhaust", "thak", "tired", "recharge"]):
        return "burnout"
    else:
        return "anxiety"


def render_visual_pathway_svg(pathway_type: str = "exam") -> str:
    """
    Renders a high-tech modern visual pathway roadmap SVG
    with connected glowing nodes and step checkpoints.
    Safe against XML parse errors with strict HTML/XML escaping.
    """
    template = PATHWAY_TEMPLATES.get(pathway_type, PATHWAY_TEMPLATES["exam"])
    title = _xml_escape(template["title"])
    desc = _xml_escape(template.get("description", "Visual Step-by-Step Navigation for MindSaathi"))
    steps = template["steps"]

    card_h = 76
    gap = 20
    total_h = 95 + len(steps) * (card_h + gap) + 30
    width = 680

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {total_h}" width="100%" height="{total_h}" style="border-radius:18px;background:linear-gradient(180deg,#0e1020 0%,#090a14 100%);border:1px solid rgba(155,89,245,0.3);margin:14px 0;box-shadow:0 8px 32px rgba(0,0,0,0.4);">',
        '<defs>',
        '  <linearGradient id="pathGrad" x1="0%" y1="0%" x2="0%" y2="100%">',
        '    <stop offset="0%" stop-color="#00F5D4"/>',
        '    <stop offset="50%" stop-color="#00BBF9"/>',
        '    <stop offset="100%" stop-color="#9B5DE5"/>',
        '  </linearGradient>',
        '  <filter id="glow">',
        '    <feGaussianBlur stdDeviation="3.5" result="coloredBlur"/>',
        '    <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>',
        '  </filter>',
        '</defs>',
        f'<rect x="2" y="2" width="{width-4}" height="{total_h-4}" rx="16" fill="rgba(255,255,255,0.015)"/>',
        f'<text x="32" y="44" font-family="Inter, system-ui, sans-serif" font-size="18" font-weight="800" fill="#FFFFFF">{title}</text>',
        f'<text x="32" y="68" font-family="Inter, system-ui, sans-serif" font-size="12" font-weight="500" fill="#94A3B8">{desc}</text>',
    ]

    line_x = 52
    line_start_y = 110
    line_end_y = 110 + (len(steps) - 1) * (card_h + gap)
    svg_parts.append(
        f'<line x1="{line_x}" y1="{line_start_y}" x2="{line_x}" y2="{line_end_y}" stroke="url(#pathGrad)" stroke-width="4" stroke-linecap="round" filter="url(#glow)"/>'
    )

    for idx, (step_title, step_desc, icon) in enumerate(steps):
        y = 110 + idx * (card_h + gap)
        st_safe = _xml_escape(step_title)
        sd_safe = _xml_escape(step_desc)

        # Node circle
        svg_parts.append(
            f'<circle cx="{line_x}" cy="{y + card_h//2}" r="15" fill="#0e1020" stroke="#00F5D4" stroke-width="3" filter="url(#glow)"/>'
        )
        svg_parts.append(
            f'<text x="{line_x}" y="{y + card_h//2 + 5}" font-family="Inter, system-ui, sans-serif" font-size="11" font-weight="bold" fill="#00F5D4" text-anchor="middle">{idx+1}</text>'
        )

        card_x = line_x + 32
        card_w = width - card_x - 30
        svg_parts.append(
            f'<rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" rx="12" fill="rgba(255,255,255,0.04)" stroke="rgba(155,89,245,0.22)"/>'
        )
        svg_parts.append(
            f'<text x="{card_x + 18}" y="{y + 28}" font-family="Inter, system-ui, sans-serif" font-size="14" font-weight="700" fill="#E2D9F9">{icon} {st_safe}</text>'
        )
        svg_parts.append(
            f'<text x="{card_x + 18}" y="{y + 52}" font-family="Inter, system-ui, sans-serif" font-size="12" font-weight="400" fill="#94A3B8">{sd_safe}</text>'
        )

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def get_all_pathways() -> Dict[str, Dict]:
    """Returns all available pathway templates."""
    return PATHWAY_TEMPLATES

