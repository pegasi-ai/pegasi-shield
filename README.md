# 🛡️🔗 GuardChain
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![plot](./static/images/guardrail_img.png)

GuardChain is a lightweight toolkit for fine-tuning and deploying powerful, safe, and customized large language models. Behind the scenes, we're building an enterprise-grade LangChain alternative to scale responsible AI applications, starting with HuggingFace GGML models.

Our toolkit accelerates the time-to-production of custom LLMs by transforming unstructured data to `.json` for fine-tuning and capturing responsible AI metrics of outputs/prompts to mitigate risks and improve performance. 

## Quickstart 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KCn1HIeD3fQy8ecT74yHa3xgJZvdNvqL?usp=sharing)

Get started with the below in minutes via a free colab instance: 
1. Automated Evaluations of LLM for Text Quality, Toxicity, Bias, Relevance, Sentiment, Prompt Injection, etc.
2. Simple Generation of JSON-formatted Q&A dataset from your data (PDFs) via open-source LLMs for fine-tuning (SFT)
3. Auditable Logs of evaluation metrics for observability, benchmarking, and compliance

## Installation 💻

To install guardrail-ml, use the Python Package Index (PyPI) as follows:

```
pip install guardrail-ml
```

## Features
Guardrail ML supports the following `metrics` and logs them:
- Toxicity & Bias
- Text Quality
- Text Relevance
- Privacy
- Sentiment

Guardrail ML can transform your data from:
- PDFs into `.json` question & answer pairs
- Uses `dolly-v2` as default to generate pairs
- Leverage your huggingface models to generate pairs

View logs in `streamlit` dashboard
- Locally deployed dashboard to view metrics
- Be used for auditing  benchmarking experiments

## Usage
```python
from guardrail.client import run_metrics
from guardrail.client import run_simple_metrics
from guardrail.client import create_dataset

# Output/Prompt Metrics
run_metrics(output="Guardrail is an open-source toolkit for building domain-specific language models with confidence. From domain-specific dataset creation and custom     evaluations to safeguarding and redteaming aligned with policies, our tools accelerates your LLM workflows to systematically derisk deployment.",
            prompt="What is guardrail-ml?",
            model_uri="dolly-v2-0.01")

# View Logs
con = sqlite3.connect("logs.db")
df = pd.read_sql_query("SELECT * from logs", con)
df.tail(20)

# Generate Dataset from PDF
create_dataset(model="OpenAssistant/falcon-7b-sft-mix-2000",
               tokenizer="OpenAssistant/falcon-7b-sft-mix-2000",
               file_path="example-docs/Medicare Appeals Paper FINAL.pdf",
               output_path="./output.json",
               load_in_4bit=True)
```

## More Colab Notebooks
4-bit QLoRA of `llama-v2-7b` with `dolly-15k` (07/21/23): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing)

Fine-Tuning Dolly 2.0 with LoRA: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n5U13L0Bzhs32QO_bls5jwuZR62GPSwE?usp=sharing)

Inferencing Dolly 2.0: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A8Prplbjr16hy9eGfWd3-r34FOuccB2c?usp=sharing)

### Related AI Papers:
- [Universal and Transferable Adversarial Attacks
on Aligned Language Models](https://llm-attacks.org/zou2023universal.pdf)
- [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v09.pdf)
