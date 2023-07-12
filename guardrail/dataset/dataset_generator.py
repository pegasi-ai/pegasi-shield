import re
import json
import csv

from typing import List, Union, Dict, Any

from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from jsonformer.format import highlight_values
from jsonformer.main import Jsonformer

from unstructured.partition.auto import partition
from unstructured.documents.elements import NarrativeText
from unstructured.partition.text_type import sentence_count

class DatasetGenerator:
 
    def __init__(
            self,    
            file_path: str,
            output_path: str,
             *,
            debug: bool = False,
            max_array_length: int = 256,
            max_number_tokens: int = 64,
            temperature: float = 0.3,
            max_string_token_length: int = 1024
            ):
        
        self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-2-8b",
                                          padding_side="left",
                                          use_fast=True,
                                          max_length=1024,
                                          use_cache=True)        
        self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-2-8b",
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16,
                                             use_cache=True)
        self.max_array_length = max_array_length
        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length
        self.debug = debug

        self.command = "You are an API that converts bodies of text into five highly contextual questions and answer pairs into a JSON format from the following text: "
        self.schema = " Generate five question and answer pairs based on the following schema:"
        self.json_schema = {
            "type": "object",
            "properties": {
                "qa_pair1": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    }
                },
                "qa_pair2": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    }
                },
                "qa_pair3": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    }
                },
                "qa_pair4": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    }
                },
                "qa_pair5": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    }
                }
            }
        }
        self.file_path = file_path
        self.output_path = output_path
    
    def generate_dataset(self):
        results = self.partition_file(self.file_path)
        qa_pairs = self.generate_pairs(results)
        json_output = self.convert_json(qa_pairs)
        self.validate_json(json_output)
        self.save_json(json_output, self.output_path)
    
    def partition_file(self, file_path):
        results = []
        if file_path:
            elements = partition(filename=file_path, strategy="hi_res")
            print("Elements", len(elements))
            for element in elements:
                if isinstance(element, NarrativeText) and sentence_count(element.text) > 1:
                    results.append(element.text)
        return results
    
    def generate_pairs(self, results):
        qa_pairs_arr = []
        print(len(results))
        if len(results) > 0:
            for text in results:
                prompt = self.generate_prompt(text)
                builder = Jsonformer(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            json_schema=self.json_schema,
                            prompt=prompt,
                            max_string_token_length=self.max_string_token_length,
                            max_array_length=self.max_array_length,
                            max_number_tokens=self.max_number_tokens,
                            temperature=self.temperature)
                output = builder()
                qa_pairs_arr.append(output)
                print(output)
                print("Size of array: ", len(qa_pairs_arr))
        return qa_pairs_arr
    
    def generate_prompt(self, input):
        prompt_intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        prompt_instruction = """

        ### Instruction:
        Heed the following rules:
        - Generate five highly contextual prompts and answer pairs from the following context
        - Each question must increasingly be more sophisticated and complex
        - Only return values that are explicitly mentioned in the text and match one of the provided options in the schema.
        - For each question, triple check that it is a valid question ending with ?
        - For each answer, only output answers that can be reference in the follwing text context. 

        You are an API that converts bodies of text into five (5x) unique question and answer pairs into a JSON format 
        from the following text, while sticking to the aforementioned rules:
        """
        INPUT_KEY = "Context:"
        RESPONSE_KEY = "category=closed_qa"

        prompt_schema = "Generate five different question and answer pairs sticking to the aforementioned rules and based on the following schema"
        prompt_full = """{intro}
        {instruction}
        {response_key}

        {input_key}
        {input}

        {schema}
        """.format(
            intro=prompt_intro,
            instruction=prompt_instruction,
            input_key=INPUT_KEY,
            input=input,
            response_key=RESPONSE_KEY,
            schema=prompt_schema)
        return prompt_full

    
    def convert_json(self, json_data):
        """Converts the given JSON data into a single JSON output with just the question and answer pairs as objects.

        Args:
            json_data: The JSON data to be converted.

        Returns:
            A JSON object with the question and answer pairs as objects.
        """
        qa_pairs = []
        for item in json_data:
            for qa_pair in item.keys():
                question = item[qa_pair]["question"]
                answer = item[qa_pair]["answer"]

                # Clean up the question and answer pairs for any special characters.
                question = re.sub(r"[^\w\s,.!?]", "", question)
                answer = re.sub(r"[^\w\s,.!?]", "", answer)

                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
        return json.dumps(qa_pairs)

    def validate_json(self, json_data):
        """Validates that the given JSON data is valid JSON.

        Args:
            json_data: The JSON data to be validated.

        Raises:
            Exception: If the given JSON data is not valid JSON.
        """
        try:
            json.loads(json_data)
        except Exception:
            raise Exception("The given JSON data is not valid JSON.")

    def save_json(self, json_data, filename):
        """Saves the given JSON data to a file.

        Args:
            json_data: The JSON data to be saved.
            filename: The filename to save the JSON data to.
        """
        with open(filename, "w") as f:
            json.dump(json_data.strip('"'), f)
