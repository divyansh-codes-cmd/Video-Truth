import sqlite3
from datetime import datetime

DB_FILE = "videotruth.db"

def init_db():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    # Analyses table
    c.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            video_url TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            signals TEXT,
            created_at TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

# ===== USER FUNCTIONS =====

def create_user(username, email, password_hash):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (username, email, password, created_at)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    return row

def email_exists(email):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE email = ?', (email,))
    row = c.fetchone()
    conn.close()
    return row is not None

# ===== ANALYSIS FUNCTIONS =====

def save_result(username, video_url, verdict, confidence, signals):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    c.execute('''
        INSERT INTO analyses (username, video_url, verdict, confidence, signals, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, video_url, verdict, confidence, str(signals),
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_history(username=None):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    if username:
        c.execute('SELECT * FROM analyses WHERE username = ? ORDER BY created_at DESC LIMIT 20', (username,))
    else:
        c.execute('SELECT * FROM analyses ORDER BY created_at DESC LIMIT 20')
    rows = c.fetchall()
    conn.close()
    return rows

def get_stats(username=None):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    if username:
        c.execute('SELECT COUNT(*) FROM analyses WHERE username = ?', (username,))
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM analyses WHERE verdict = 'AI Generated' AND username = ?", (username,))
        ai_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM analyses WHERE verdict = 'Real' AND username = ?", (username,))
        real_count = c.fetchone()[0]
    else:
        c.execute('SELECT COUNT(*) FROM analyses')
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM analyses WHERE verdict = 'AI Generated'")
        ai_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM analyses WHERE verdict = 'Real'")
        real_count = c.fetchone()[0]
    conn.close()
    return {"total": total, "ai": ai_count, "real": real_count}

