import sqlite3

def create_database():
    conn = sqlite3.connect('smart_shopping.db')
    cursor = conn.cursor()

    # Create users table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        face_embedding BLOB NOT NULL
                    )''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_database()
    print("Database and users table created successfully.")
















