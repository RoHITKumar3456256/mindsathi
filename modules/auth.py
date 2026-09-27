import os
import sqlite3
import hashlib
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "sessions.db")

def hash_password(password: str) -> str:
    """Returns SHA-256 hash of password."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_auth_db():
    """Initializes user authentication and conversation history tables."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username         TEXT UNIQUE NOT NULL,
        password_hash    TEXT NOT NULL,
        full_name        TEXT,
        stream           TEXT,
        created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # User memory table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_memory (
        memory_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id          INTEGER,
        key_topic        TEXT,
        memory_text      TEXT,
        updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)
    
    # Conversation Sessions Master table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        conv_id          TEXT PRIMARY KEY,
        user_id          INTEGER,
        title            TEXT,
        created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_updated     DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    conn.close()

def register_user(username, password, full_name, stream) -> tuple:
    """Registers a new user in SQLite DB."""
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        p_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, stream) VALUES (?, ?, ?, ?)",
            (username.lower().strip(), p_hash, full_name.strip(), stream)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return (True, {"user_id": user_id, "username": username, "full_name": full_name, "stream": stream})
    except sqlite3.IntegrityError:
        conn.close()
        return (False, "Username already exists. Please choose a different username.")
    except Exception as e:
        conn.close()
        return (False, f"Registration failed: {str(e)}")

def login_user(username, password) -> tuple:
    """Validates user credentials."""
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    p_hash = hash_password(password)
    cursor.execute(
        "SELECT user_id, username, full_name, stream FROM users WHERE username = ? AND password_hash = ?",
        (username.lower().strip(), p_hash)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return (True, {"user_id": row[0], "username": row[1], "full_name": row[2], "stream": row[3]})
    return (False, "Invalid username or password.")

def get_user_conversations(user_id: int) -> list:
    """Fetches past conversation sessions for a user."""
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT conv_id, title, last_updated FROM conversations WHERE user_id = ? ORDER BY last_updated DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"conv_id": r[0], "title": r[1], "last_updated": r[2]} for r in rows]

def create_or_update_conversation(conv_id: str, user_id: int, title: str):
    """Creates or updates a conversation title and timestamp."""
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO conversations (conv_id, user_id, title, last_updated)
    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(conv_id) DO UPDATE SET title = excluded.title, last_updated = CURRENT_TIMESTAMP
    """, (conv_id, user_id, title))
    
    conn.commit()
    conn.close()

def delete_conversation(conv_id: str):
    """Deletes a conversation and its messages."""
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (conv_id,))
    conn.commit()
    conn.close()

def get_user_memory_summary(user_id: int) -> str:
    """Summarizes user memory for Gemini context."""
    if not user_id:
        return ""
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT full_name, stream FROM users WHERE user_id = ?", (user_id,))
    u_row = cursor.fetchone()
    
    cursor.execute("""
    SELECT user_message, detected_emotion, pre_pss, post_pss FROM sessions 
    WHERE session_id IN (SELECT conv_id FROM conversations WHERE user_id = ?)
    ORDER BY id DESC LIMIT 10
    """, (user_id,))
    turns = cursor.fetchall()
    conn.close()
    
    name = u_row[0] if u_row else "Student"
    stream = u_row[1] if u_row else "College"
    
    summary = f"Student Name: {name}, Stream: {stream}. "
    if turns:
        recent_emotions = set([t[1] for t in turns if t[1]])
        summary += f"Recent emotional states: {', '.join(recent_emotions)}. "
        
    return summary
