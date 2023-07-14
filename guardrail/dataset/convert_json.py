import json
import csv

instruction = "You will answer questions about the US constitution "

with open("./responses.json", "r") as f:
    responses = json.load(f)

# Open the CSV file and write the data to it
with open("responses.csv", "w", newline="") as csvfile:
    fieldnames = ["question", "answer"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for response in responses["responses"]:
        if "question" in response and "answer" in response:  # To make sure both keys exist
            writer.writerow(
                {
                    "question": instruction + response["question"],
                    "answer": response["answer"],
                }
            )
