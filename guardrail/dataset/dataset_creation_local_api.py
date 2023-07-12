import PyPDF2
from transformers import AutoTokenizer, LlamaTokenizer, pipeline
import json
import requests
import re
import torch
import csv
from cleantext import clean
from unstructured.partition.auto import partition

# 1. Change Tokenizer
# 2. Change system_template
# 3. Change PDF file

class DatasetCreator:
 
    def __init__(self, file_path=None, model=None, model_endpoint=None, model_pipeline=None, tokenizer=None, prompt=None, output_filetype=None, system=None, instruction=None):
        self.file_path = file_path
        self.model = model
        self.model_endpoint = model_endpoint,
        self.model_pipeline = model_pipeline,
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.output_filetype = output_filetype
        self.system = system
        self.instruction = instruction
        self.text_gen = pipeline(model=self.model,
                                tokenizer=self.tokenizer,
                                torch_dtype=torch.bfloat16, 
                                trust_remote_code=True,
                                load_in_8bit=True,
                                device_map="auto")  
    def create_dataset(self):
        self.initialize_vars()
        token = False
        if self.model_pipeline: 
            chunks = self.split_text(self.file_path)
        else:
            chunks = self.tokenize_file(self.file_path)
            token = True
        responses = self.get_responses(chunks, token, []) 
        self.convert_json_output(responses)
        print("Completed creating output file")    

    def initialize_vars(self):
        if self.file_path is None or self.model is None or self.tokenizer is None or self.output_filetype is None:
            print("Missing required arguments.")
            return
        
        if self.model is None and self.model_endpoint is None:
            print("Missing either model or endpoint to run inference")
            return

        if self.prompt is None:
            self.prompt = "You are an API that converts bodies of text into a single question and answer into a JSON format. Each JSON " \
          "contains a single question with a single answer. Only respond with the JSON and no additional text. \n"

        if self.system is None: 
            self.system = 'You are an AI assistant that follows system extremely well. Help as much as you can.'
        
        if self.output_filetype is None:
            self.output_filetype = "csv"

        if self.instruction is None:
            self.instruction = "You will answer questions about the files"  
         

    # TODO support local oobabooga inference API
    def generate_qa_pair(self, system, prompt, input):
        if input:
            # prompt = f"### System:\n{system}\n\n### User:\n{prompt}\n\n### Input:\n{input}\n\n### Response:\n"
            prompt = self.prompt + input
        else:
            prompt = f"### System:\n{system}\n\n### User:\n{prompt}\n\n### Response:\n"

        if self.text_gen:   
            instance = {'top_p': 0.3, 'temperature':0.4, 'generate_len': 256, 'top_k': 10}
   
            output = self.text_gen(prompt, 
                              top_k=instance['top_k'],
                              use_cache=True,
                              do_sample=True,
                              top_p=instance['top_p'],
                              temperature=instance['temperature'])
            
            return output[0]['generated_text'].replace("\n", "")

        else:
            instance = {'input_ids': tokens,'top_p': 0.2, 'temperature':0.3, 'generate_len': 256, 'top_k': 3}

            tokens = self.tokenizer.encode(prompt)
            tokens = torch.LongTensor(tokens).unsqueeze(0)
            tokens = tokens.to('cuda')

            length = len(tokens[0])
            
            with torch.no_grad():
                rest = self.model.generate(
                    input_ids=tokens,
                    max_length=length+instance['generate_len'],
                    use_cache=True,
                    do_sample=True,
                    top_p=instance['top_p'],
                    temperature=instance['temperature'],
                    top_k=instance['top_k']
                )
            output = rest[0][length:]
            string = self.tokenizer.decode(output, skip_special_tokens=True)
            return f'[!] Response: {string}'

    '''
    def extract_text_from_pdf(self, file_path):
        pdf_file_obj = open(file_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[page_num]
            text += page_obj.extract_text()
        pdf_file_obj.close()
        cleaned_text = clean(text, lower=False)
        return cleaned_text
    '''
    
    # TODO: PDF, HTML, DOCX
    def extract_text_from_unstructured(self, file_path):
        elements = partition(filename=file_path)
        text = "\n\n".join([str(el) for el in elements])
        cleaned_text = clean(text, lower=False)
        return cleaned_text

    def tokenize_pdf(self, path):
        text = self.extract_text_from_unstructured(path)
        tokens = self.tokenize(text)

        token_chunks = list(self.chunks(tokens, 256))
        print("Number of token_chunks", len(token_chunks))
        return token_chunks

    def tokenize(self, text):
        enc = self.tokenizer.encode(text)
        return enc

    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def is_json(self, data):
        try:
            json_data =json.loads(data)
            if "question" in json_data and "answer" in json_data:
                return True
        except ValueError:
            return False
        
    def convert_to_json(self, output):
        # Remove leading/trailing whitespaces and newlines
        output = output.strip()

        # Initialize an empty JSON object
        json_object = {}

        # Extract the JSON string from the output
        json_start = output.find("{")
        json_end = output.rfind("}")
        json_string = output[json_start:json_end + 1]

        # Try parsing the JSON string
        try:
            json_data = json.loads(json_string)
            if "question" in json_data and "answer" in json_data:
                json_object["question"] = json_data["question"]
                json_object["answer"] = json_data["answer"]
        except ValueError:
            # Extract question and answer from non-JSON format
            lines = output.splitlines()
            for line in lines:
                line = line.strip()
                if line.startswith("Response:") or line.startswith("\"response\"") or line.startswith("[!] Response:"):
                    line = line.split(":", 1)[1].strip().strip("\"").strip()
                if line.startswith("Question:") or line.startswith("\"question\"") or line.startswith("question:"):
                    question = line.split(":", 1)[1].strip().strip("\"").strip()
                    json_object["question"] = question
                elif line.startswith("Answer:") or line.startswith("\"answer\""):
                    answer = line.split(":", 1)[1].strip().strip("\"").strip()
                    json_object["answer"] = answer

        return json.dumps(json_object)
    
    def submit_to_api(self, chunk, retries=5):
        for i in range(retries):
            try:
                if i > 1:
                    initial_response = self.generate_qa_pair(self.system, self.prompt, chunk.strip())
                initial_response = self.generate_qa_pair(self.system, self.prompt, chunk.strip())
                print("Initial response", initial_response)
                # Extract JSON string from between back-ticks
                response = self.convert_to_json(initial_response)
                print("cleaned up response", response)
                if self.is_json(response):
                    print(response)
                    return json.loads(response)
                else:
                    match = re.search(r'`(.*?)`', response, re.S)
                    if match and self.is_json(match.group(1)):
                        print(f"Attempt {i + 1} failed. Retrying...")
                        return json.loads(match.group(1))  # assuming you want to return the JSON data
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                continue
        print("Max retries exceeded. Skipping this chunk.")
        return None
    
    def split_text(self, path):
        text = self.extract_text_from_unstructured(path)
        token_chunks = list(self.chunks(text, 256))
        print("Number of text chunks", len(token_chunks))
        return token_chunks
    
    def tokenize_file(self, path):
        text = self.extract_text_from_unstructured(path)
        tokens = self.tokenize(text)

        token_chunks = list(self.chunks(tokens, 256))
        print("Number of token_chunks", len(token_chunks))
        return token_chunks
    
    def get_responses(self, token_chunks, token, responses):
        for chunk in token_chunks:
            if token: response = self.submit_to_api(self.tokenizer.decode(chunk))
            else: response = self.submit_to_api(chunk)
            if response is not None:
                responses.append(response)
            print("Size of responses", len(responses))

        # Write responses to a JSON file
        with open('responses.json', 'w') as f:
            json.dump(responses, f)

    def convert_json_output(self, responses):
        # Open the CSV file and write the data to it
        with open('responses.csv', 'w', newline='') as csvfile:
            fieldnames = ['question', 'answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for response in responses['responses']:
                if 'question' in response and 'answer' in response:  # To make sure both keys exist
                    writer.writerow({'question': self.instruction + response['question'], 'answer': response['answer']})