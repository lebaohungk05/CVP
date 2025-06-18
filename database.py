import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

def init_db():
    conn = sqlite3.connect('emotion_app.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create emotion_logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS emotion_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Create demo user if not exists
    create_demo_user()

def create_demo_user():
    """Create a demo user for testing"""
    try:
        add_user('demo', 'demo123', 'demo@example.com')
        print("✅ Demo user created: username='demo', password='demo123'")
    except:
        print("ℹ️ Demo user already exists")

def add_user(username, password, email):
    conn = sqlite3.connect('emotion_app.db')
    c = conn.cursor()
    try:
        hashed_password = generate_password_hash(password)
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                 (username, hashed_password, email))
        conn.commit()
        print(f"✅ User '{username}' created successfully")
        return True
    except sqlite3.IntegrityError as e:
        print(f"❌ Error creating user '{username}': {e}")
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('emotion_app.db')
    c = conn.cursor()
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        print(f"✅ Login successful for user '{username}'")
        return user[0]  # Return user_id
    else:
        print(f"❌ Login failed for user '{username}'")
        return None

def log_emotion(user_id, emotion, confidence):
    conn = sqlite3.connect('emotion_app.db')
    c = conn.cursor()
    c.execute('INSERT INTO emotion_logs (user_id, emotion, confidence) VALUES (?, ?, ?)',
             (user_id, emotion, confidence))
    conn.commit()
    conn.close()

def get_user_emotion_history(user_id, limit=50):
    conn = sqlite3.connect('emotion_app.db')
    c = conn.cursor()
    c.execute('''
        SELECT user_id, emotion, confidence, timestamp 
        FROM emotion_logs 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (user_id, limit))
    history = c.fetchall()
    conn.close()
    return history

def get_all_users():
    """Get all users for debugging"""
    conn = sqlite3.connect('emotion_app.db')
    c = conn.cursor()
    c.execute('SELECT id, username, email, created_at FROM users')
    users = c.fetchall()
    conn.close()
    return users

def reset_database():
    """Reset database for testing"""
    import os
    if os.path.exists('emotion_app.db'):
        os.remove('emotion_app.db')
        print("🗑️ Database reset")
    init_db()
    print("🆕 New database created with demo user") 