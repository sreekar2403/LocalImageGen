import os
import gc
import subprocess

# [optional] Force Ollama to run on CPU to avoid GPU memory pressure
os.environ["OLLAMA_GPU"] = "0"

import ollama
import torch

from config import ollama_model
from utils import SYSTEM_PROMPTS

def stop_ollama_model(ollama_model):
    try:
        subprocess.run(
            ["ollama", "stop", ollama_model],
            check=True
        )

        print(f"{ollama_model} unloaded from memory")

    except subprocess.CalledProcessError:
        print("Could not stop model")

def generate_prompt(user_prompt):
    for _, key in enumerate(SYSTEM_PROMPTS.keys(), start=1):
        print(f"[{_}] : {key}")
    option_chose = input("Enter option: ")
    print(SYSTEM_PROMPTS.get(option_chose, SYSTEM_PROMPTS["flux"]))
    response = ollama.chat(
        model=ollama_model,
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPTS.get(
                    option_chose,
                    SYSTEM_PROMPTS["flux"]
                )
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
    )
    try:
        print(f"GPU Memory used: {torch.cuda.memory_allocated() / 1024**2} MB")
        stop_ollama_model(ollama_model)
        print("ollama model offloaded")
        print(f"GPU Memory freed: {torch.cuda.memory_allocated() / 1024**2} MB")
        
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        raise Exception("Ollama model not stopped")
    
    return response['message']['content']