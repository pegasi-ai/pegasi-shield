"""
1. run_metrics
2. create_dataset 
3. init_logs 
"""

import logging
import sqlite3

from .metrics.textstat import Textstat
from .metrics.toxicity import Toxicity
from .metrics.relevance import Relevance
from .metrics.injections import PromptInjection
from .metrics.sentiment import Sentiment
from .metrics.bias import Bias

from .dataset.dataset_generator import DatasetGenerator
from .db import insert_log

def run_simple_metrics(text, model_id):
    # Initialize the Textstat class
    textstat = Textstat()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Textstat")

    results = textstat.evaluate(text)

    for result in results:
        try:
            # Insert log into the database
            insert_log(result, results[result], model_id)
        except Exception as e:
            logger.error(f"Error while inserting {result} into DB: {e}")

            # Insert error log into the database
            insert_log(result,  f"Error: {e}", model_id)

    return results

def run_metrics(text, prompt=None, model_id=-1):
    # Initialize the metrics classes
    textstat = Textstat()
    toxicity = Toxicity()
    relevance = Relevance()
    injection = PromptInjection()
    sentiment = Sentiment()
    bias = Bias()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Textstat")

    results = {}
    ts_results = textstat.evaluate(text)
    sentiment_results = sentiment.evaluate(text)
    bias_results = bias.evaluate(text)

    if prompt:
        relevance_results = relevance.evaluate(text, prompt)
        injection_results = injection.evaluate(prompt)
        toxicity_results = toxicity.evaluate(text, prompt)

    for result in ts_results:
        try:
            # Insert log into the database
            insert_log(result, ts_results[result], model_id)
        except Exception as e:
            logger.error(f"Error while inserting {result} into DB: {e}")

            # Insert error log into the database
            insert_log(result,  f"Error: {e}", model_id)

    insert_log("toxicity", toxicity_results, model_id)

    results["text_quality"] = ts_results
    results["toxicity"] = toxicity_results
    results["sentiment"] = sentiment_results
    results["bias"] = bias_results

    if relevance_results: results["relevance"] = relevance_results
    if injection_results: results["prompt_injection"] = injection_results

    return results


def create_dataset(file_path=None, output_path="./output.json"):
    dc = DatasetGenerator(file_path=file_path, 
                          output_path=output_path)
    dc.generate_dataset()

def init_logs():
    # Create a database connection and table for logs
    conn = sqlite3.connect("logs.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            levelname TEXT,
            message TEXT,
            model_id TEXT
        )
    """)
    conn.close()
# Initialize the database
init_logs()

# Example usage
# text = "This flight is so long and unpleasant"
# model_id = "12345"  # Replace with your ML model ID
# run_metrics(text, model_id)