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


def run_simple_metrics(output, prompt, model_uri):
    # Initialize the Textstat class
    textstat = Textstat()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Textstat")

    ts_eval_results = textstat.evaluate(output)

    for result_name in ts_eval_results:
        try:
            # Insert log into the database
            insert_log(model_uri, prompt, output, result_name, ts_eval_results[result_name])
        except Exception as e:
            logger.error(f"Error while inserting {result_name} into DB: {e}")

    return ts_eval_results


def run_metrics(output, prompt, model_uri):
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
    ts_results = textstat.evaluate(output)
    sentiment_results = sentiment.evaluate(output)
    bias_results = bias.evaluate(output)

    if prompt:
        relevance_results = relevance.evaluate(output, prompt)
        injection_results = injection.evaluate(prompt)
        toxicity_results = toxicity.evaluate(output, prompt)

    results["text_quality"] = ts_results
    for result_name in ts_results:
        try:
            # Insert log into the database
            insert_log(
                model_uri,
                prompt,
                output,
                "tq_" + str(result_name),
                ts_results[result_name],
            )
        except Exception as e:
            logger.error(f"Error while inserting {result_name} into DB: {e}")

    insert_log(model_uri, prompt, output, "toxicity", toxicity_results)
    insert_log(model_uri, prompt, output, "sentiment", sentiment_results)
    insert_log(model_uri, prompt, output, "bias_label", bias_results[0]["label"])
    insert_log(model_uri, prompt, output, "bias_score", bias_results[0]["score"])

    results["toxicity"] = toxicity_results
    results["sentiment"] = sentiment_results
    results["bias"] = bias_results

    if relevance_results:
        results["relevance"] = relevance_results
        insert_log(model_uri, prompt, output, "relevance", relevance_results)
    if injection_results:
        results["prompt_injection"] = injection_results
        insert_log(model_uri, prompt, output, "prompt_injection", injection_results)
    return results


def create_dataset(
    file_path, model, tokenizer, output_path="./output.json", load_in_4bit=True, temperature=0.3
):
    dc = DatasetGenerator(
        file_path=file_path,
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        load_in_4bit=load_in_4bit,
        temperature=temperature,
    )
    dc.generate_dataset()


def init_logs():
    # Create a database connection and table for logs
    conn = sqlite3.connect("logs.db")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            model_uri TEXT,
            prompt TEXT,
            output TEXT,
            metric_name TEXT,
            metric_value TEXT
        );
    """
    )
    conn.close()


# Initialize the database
init_logs()
