import sqlite3

def migrate():
    conn = sqlite3.connect('./data/fitdesi.db')
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE daily_logs ADD COLUMN created_at DATETIME;")
        print("Migrated daily_logs.")
    except Exception as e:
         print(e)
    try:
        # For ChatMessage, created_at might already exist but without timezone
        pass
    except Exception as e:
         print(e)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    migrate()
