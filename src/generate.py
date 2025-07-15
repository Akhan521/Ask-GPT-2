from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers import GenerationConfig
import torch

def configure_generation(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> GenerationConfig:
    '''Configures generation settings for the model.'''
    generation_config = GenerationConfig.from_pretrained(model.name_or_path)

    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.max_new_tokens = 1024 # GPT-2 has a context length of 1024.
    generation_config.do_sample = True
    generation_config.temperature = 0.7     # Temperature controls randomness in generation.
    generation_config.top_p = 0.9           # Top-p sampling to limit the number of tokens considered.
    generation_config.top_k = 20            # Top-k sampling to limit the number of highest probability tokens.

    return generation_config

def generate(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device) -> str:
    '''Generates text from a prompt using the specified model and tokenizer.'''
    model.to(device)
    model.eval()

    generation_config = configure_generation(model, tokenizer)
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt').to(device)
    out = model.generate(input_ids=encoded_prompt, generation_config=generation_config, repetition_penalty=2.0, do_sample=True)
    decoded_string = tokenizer.decode(out[0].tolist(), clean_up_tokenization_spaces=True)
    print(decoded_string)
    
    return decoded_string