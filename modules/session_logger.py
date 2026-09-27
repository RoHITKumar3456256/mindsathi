import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "sessions.db")
EXPORT_DIR = os.path.join(DB_DIR, "export")


def init_db():
    """Initializes SQLite database and creates sessions table if not present."""
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id           TEXT NOT NULL,
        timestamp            DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_message         TEXT,
        detected_emotion     TEXT,
        emotion_conf         REAL,
        self_emotion         TEXT,
        bot_response         TEXT,
        pre_pss              INTEGER,
        post_pss             INTEGER,
        helpfulness          INTEGER,
        would_use_again      TEXT,
        gender               TEXT,
        stream               TEXT,
        city_tier            TEXT,
        crisis_flag          INTEGER DEFAULT 0,
        session_duration_sec INTEGER DEFAULT 0
    );
    """)
    conn.commit()
    conn.close()


def log_turn(session_id: str, user_message: str, detected_emotion: str, 
             emotion_conf: float, bot_response: str, pre_pss: int = None, 
             crisis_flag: int = 0) -> int:
    """Logs a single conversation turn into SQLite."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO sessions (
            session_id, user_message, detected_emotion, emotion_conf, 
            bot_response, pre_pss, crisis_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (session_id, user_message, detected_emotion, emotion_conf, bot_response, pre_pss, crisis_flag))
    
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_post_session(session_id: str, post_pss: int, helpfulness: int, 
                        would_use_again: str, self_emotion: str, gender: str, 
                        stream: str, city_tier: str, session_duration_sec: int = 0):
    """Updates all rows associated with a session ID with post-survey data."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE sessions 
        SET post_pss = ?,
            helpfulness = ?,
            would_use_again = ?,
            self_emotion = ?,
            gender = ?,
            stream = ?,
            city_tier = ?,
            session_duration_sec = ?
        WHERE session_id = ?
    """, (post_pss, helpfulness, would_use_again, self_emotion, gender, stream, city_tier, session_duration_sec, session_id))
    
    conn.commit()
    conn.close()


def export_csv(output_path: str = None) -> str:
    """Exports SQLite sessions table to CSV for SPSS/JASP statistical analysis."""
    init_db()
    if output_path is None:
        output_path = os.path.join(EXPORT_DIR, f"mindsathi_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sessions", conn)
    conn.close()
    
    # Compute derived variables for SPSS
    if not df.empty and "pre_pss" in df.columns and "post_pss" in df.columns:
        df["stress_change"] = df["post_pss"] - df["pre_pss"]
        if "detected_emotion" in df.columns and "self_emotion" in df.columns:
            df["emotion_match"] = (df["detected_emotion"] == df["self_emotion"]).astype(int)
            
    df.to_csv(output_path, index=False)
    return output_path


def get_all_sessions_df() -> pd.DataFrame:
    """Returns all session data as a pandas DataFrame."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sessions", conn)
    conn.close()
    return df
