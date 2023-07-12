import sqlite3

def insert_log(metric, message, model_id):
    # Create a database connection
    conn = sqlite3.connect("logs.db")

    # Insert the log into the database
    conn.execute("INSERT INTO logs (timestamp, levelname, message, model_id) VALUES (datetime('now'), ?, ?, ?)",
                 (metric, message, model_id))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
