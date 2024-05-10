from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sql_lora_path = snapshot_download(repo_id="lakshay/mistral-combined")
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", enable_lora=True)
