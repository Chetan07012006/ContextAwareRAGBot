import sqlite3

# Connect to SQLite database (creates it if it doesn't exist)
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Create users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    membership_tier TEXT NOT NULL
)
""")

# Remove existing records (optional)
cursor.execute("DELETE FROM users")

# Sample users
users = [
    (101, "Riya Sharma", "Gold"),
    (102, "Aman Verma", "Silver"),
    (103, "Neha Iyer", "Platinum")
]

cursor.executemany(
    "INSERT INTO users (user_id, name, membership_tier) VALUES (?, ?, ?)",
    users
)

conn.commit()
conn.close()

print("users.db created successfully!")