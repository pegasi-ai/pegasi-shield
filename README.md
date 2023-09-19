# 🛡️Guardrail ML
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - Python Version](https://img.shields.io/pypi/v/llm-guard)](https://pypi.org/project/guardrail-ml)
[![Downloads](https://static.pepy.tech/badge/guardrail-ml)](https://pepy.tech/project/guardrail-ml)
[![Downloads](https://static.pepy.tech/badge/guardrail-ml/month)](https://pepy.tech/project/guardrail-ml)

![plot](./static/images/guardrail_img.png)

Guardrail ML is an alignment toolkit to use LLMs safely and securely. Our firewall scans prompts and LLM behaviors for risks to bring your AI app from prototype to production with confidence.

## Benefits
- 🚀mitigate LLM security and safety risks 
- 📝customize and ensure LLM behaviors are safe and secure
- 💸monitor incidents, costs, and responsible AI metrics 

## Features 
- 🛠️ firewall that safeguards against CVEs and improves with each attack
- 🤖 reduce and measure ungrounded additions (hallucinations) with tools
- 🛡️ multi-layered defense with heuristic detectors, LLM-based, vector DB

## Quickstart 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KCn1HIeD3fQy8ecT74yHa3xgJZvdNvqL?usp=sharing)

## Installation 💻
1. Get API Key

2. To install guardchain, use the Python Package Index (PyPI) as follows:

```
pip install guardrail-ml
```

## Usage 🛡️🔗

```python
from guardrail.client import run_metrics
from guardrail.client import run_simple_metrics
from guardrail.client import create_dataset

# Output/Prompt Metrics
run_metrics(output="Guardrail is an open-source toolkit for building domain-specific language models with confidence. From domain-specific dataset creation and custom     evaluations to safeguarding and redteaming aligned with policies, our tools accelerates your LLM workflows to systematically derisk deployment.",
            prompt="What is guardrail-ml?",
            model_uri="llama-v2-guanaco")

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
