from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple

def load_model(model_name_or_path: str) -> AutoModelForCausalLM:
    '''Loads a pre-trained model from the Hugging Face Hub.'''
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        
    model.config.use_cache = True  # Enable caching for faster inference.
    return model
