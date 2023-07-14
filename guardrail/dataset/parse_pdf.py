import PyPDF2
from transformers import AutoTokenizer, LlamaTokenizer
import json
import requests
import re

# 1. Change Tokenizer
# 2. Change instruction_template
# 3. Change PDF file

HOST = "localhost:5000"
URI = f"http://{HOST}/api/v1/chat"
tokenizer = LlamaTokenizer.from_pretrained("psmathur/orca_mini_v2_7b")
history = {"internal": [], "visible": []}
command = (
    "You are an API that converts bodies of text into a single question and answer into a JSON format. Each JSON "
    "contains a single question with a single answer that is referenceable. Only respond with the JSON and no additional text. \n"
)


def run(user_input, history):
    request = {
        "user_input": user_input,
        "history": history,
        "mode": "instruct",  # Valid options: 'chat', 'chat-instruct', 'instruct'
        "character": "Example",
        "instruction_template": "Orca Mini",
        "your_name": "Kevin",
        "regenerate": False,
        "_continue": False,
        "stop_at_newline": False,
        "chat_prompt_size": 2048,
        "chat_generation_attempts": 1,
        "chat-instruct_command": "",
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.1,
        "typical_p": 1,
        "epsilon_cutoff": 0,  # In units of 1e-4
        "eta_cutoff": 0,  # In units of 1e-4
        "tfs": 1,
        "top_a": 0,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 500,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }

    response = requests.post(URI, json=request)

    result = response.json()["results"][0]["history"]
    return result["visible"][-1][1]


def extract_text_from_pdf(file_path):
    pdf_file_obj = open(file_path, "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    pdf_file_obj.close()
    return text


def tokenize(text):
    enc = tokenizer.encode(text)
    return enc


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def is_json(data):
    try:
        json.loads(data)
        return True
    except ValueError:
        return False


def submit_to_api(chunk, retries=3):
    for i in range(retries):
        try:
            response = run(command + chunk.strip(), history)
            # Extract JSON string from between back-ticks
            print("initial response", response)
            if is_json(response):
                print(response)
                return json.loads(response)
            else:
                match = re.search(r"`(.*?)`", response, re.S)
                if match and is_json(match.group(1)):
                    print(f"Attempt {i + 1} failed. Retrying...")
                    return json.loads(match.group(1))  # assuming you want to return the JSON data
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            continue
    print("Max retries exceeded. Skipping this chunk.")
    return None


def tokenize_pdf(path):
    text = extract_text_from_pdf(path)
    tokens = tokenize(text)

    token_chunks = list(chunks(tokens, 256))
    print("Number of token_chunks", len(token_chunks))
    return token_chunks


def get_responses(token_chunks, responses):
    for chunk in token_chunks:
        response = submit_to_api(tokenizer.decode(chunk))
        if response is not None:
            responses.append(response)

    # Write responses to a JSON file
    with open("responses.json", "w") as f:
        json.dump(responses, f)


token_chunks = tokenize_pdf("./constitution.pdf")
responses = get_responses(token_chunks, [])
