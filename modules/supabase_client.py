import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-supabase-anon-key")

_supabase_client = None

def get_supabase():
    """Initializes and returns Supabase client if configured."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
        
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if url and key and url != "https://your-project.supabase.co":
        try:
            from supabase import create_client
            _supabase_client = create_client(url, key)
            return _supabase_client
        except Exception:
            pass
    return None

def sign_in_with_oauth(provider: str) -> dict:
    """
    Returns OAuth authorization URL for Google / GitHub / Gmail OAuth.
    """
    client = get_supabase()
    if client:
        try:
            res = client.auth.sign_in_with_oauth({"provider": provider})
            return {"status": "success", "url": res.url}
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    # Mock OAuth authorization URL if Supabase credentials are in setup mode
    mock_urls = {
        "google": f"{SUPABASE_URL}/auth/v1/authorize?provider=google",
        "github": f"{SUPABASE_URL}/auth/v1/authorize?provider=github",
        "email": f"{SUPABASE_URL}/auth/v1/authorize?provider=email"
    }
    return {"status": "success", "url": mock_urls.get(provider, SUPABASE_URL)}


def send_welcome_email_notification(to_email: str, user_name: str) -> bool:
    """
    Sends a welcome email notification to the user from MindSaathi.
    """
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    
    if not smtp_user or not smtp_pass:
        # Simulated logged notification if SMTP credentials are in template mode
        print(f"[EMAIL NOTIFICATION TRIGGERED] Sent welcome email to {to_email} for user {user_name}")
        return True
        
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🧠 Welcome to MindSaathi AI — Your Mental Health Companion!"
        msg["From"] = f"MindSaathi AI <{smtp_user}>"
        msg["To"] = to_email
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #0F0C20; color: #F3E8FF; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: #191433; border: 1px solid #9333EA; border-radius: 16px; padding: 30px;">
                <h1 style="color: #C084FC;">Welcome to MindSaathi, {user_name}! 👋</h1>
                <p style="font-size: 1.05rem; line-height: 1.6; color: #E9D5FF;">
                    Thank you for signing in to <b>MindSaathi AI</b>. We are here to support your mental health, 
                    academic stress, exam anxiety, and emotional wellbeing 24/7 with complete confidentiality.
                </p>
                <div style="background: rgba(147, 51, 234, 0.2); border-left: 4px solid #EC4899; padding: 15px; margin: 20px 0; border-radius: 8px;">
                    🤖 <i>"Need our help now? Ask me any question about your mind, study pressure, or feelings."</i>
                </div>
                <p style="color: #A78BFA; font-size: 0.9rem;">
                    With care,<br>
                    <b>The MindSaathi AI Team</b>
                </p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
