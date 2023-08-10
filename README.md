# 🛡️🔗 GuardChain
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![plot](./static/images/guardrail_img.png)

Guardchain is a lightweight toolkit for developers to evaluate, benchmark, and safeguard your AI agents and chains.

## Features 
- 🛠️ evaluate and track prompts and LLM outputs with automated text and NLP metrics 
- 🤖 benchmark domain-specific tasks with automated agent simulated conversations
- 🛡️ safeguard LLMs with our customizable firewall and enforce policies with guardrails 

## Quickstart 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KCn1HIeD3fQy8ecT74yHa3xgJZvdNvqL?usp=sharing)

## Installation 💻

To install guardchain, use the Python Package Index (PyPI) as follows:

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

### Related AI Papers & Resources:
- [Universal and Transferable Adversarial Attacks
on Aligned Language Models](https://llm-attacks.org/zou2023universal.pdf)
- [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v09.pdf)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [FacTool: Factuality Detection in Generative AI](https://arxiv.org/abs/2307.13528)
