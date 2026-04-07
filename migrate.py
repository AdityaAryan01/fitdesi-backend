import sqlite3
import os

db_path = './data/fitdesi.db'
if not os.path.exists(db_path):
    print("Database not found!")
    exit(1)

conn = sqlite3.connect(db_path)
try:
    try:
        conn.execute('ALTER TABLE users ADD COLUMN gender VARCHAR DEFAULT "male"')
    except Exception as e:
        pass
    
    try:
        conn.execute('ALTER TABLE users ADD COLUMN activity_level VARCHAR DEFAULT "moderate"')
    except Exception as e:
        pass

    try:
        conn.execute('ALTER TABLE users ADD COLUMN active_tracking_date DATE')
        conn.execute('ALTER TABLE users ADD COLUMN active_tracking_start DATETIME')
        print("Successfully added new tracking columns to users table.")
    except Exception as e:
        print(f"Tracking columns might already exist: {e}")
        
    conn.commit()
    print("Done migrating users table.")
finally:
    conn.close()
