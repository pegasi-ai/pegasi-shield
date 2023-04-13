!pip -q install datasets loralib sentencepiece
!pip -q install git+https://github.com/huggingface/transformers # need to install from github
!pip -q install git+https://github.com/huggingface/peft.git
!pip -q install bitsandbytes
!pip install torch

import torch
from transformers import pipeline

instruct_pipeline = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

instruct_pipeline("Explain to me the difference between nuclear fission and fusion.")

instruct_pipeline("Explain self-attention and transformer to me like I'm 5 years old (ELI5).")
