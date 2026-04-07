import sqlite3
import os

db_path = './data/fitdesi.db'
if not os.path.exists(db_path):
    print("Database not found!")
    exit(1)

conn = sqlite3.connect(db_path)
try:
    conn.execute('ALTER TABLE users ADD COLUMN gender VARCHAR DEFAULT "male"')
    conn.execute('ALTER TABLE users ADD COLUMN activity_level VARCHAR DEFAULT "moderate"')
    conn.commit()
    print("Successfully added columns to users table.")
except Exception as e:
    print(f"Error (might already exist): {e}")
finally:
    conn.close()
