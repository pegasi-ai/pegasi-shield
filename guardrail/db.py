import sqlite3

def insert_log(model_uri, prompt, output, metric_name, metric_value):
    # Create a database connection
    conn = sqlite3.connect("logs.db")

    # Insert the log into the database
    conn.execute("INSERT INTO logs (timestamp, model_uri, prompt, output, metric_name, metric_value) VALUES (datetime('now'), ?, ?, ?, ?, ?)",
                 (model_uri, prompt, output, metric_name, metric_value))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
