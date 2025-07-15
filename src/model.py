from transformers import AutoModelForCausalLM

def load_model(model_name: str) -> AutoModelForCausalLM:
    '''Loads a pre-trained model from the Hugging Face Hub.'''
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    model.config.use_cache = True  # Enable caching for faster inference.
    return model