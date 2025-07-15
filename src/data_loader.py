from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_data(dataset_name: str) -> DatasetDict:
    '''Load a dataset from the Hugging Face Hub.'''
    dataset = load_dataset(dataset_name)
    return dataset

def get_tokenizer(model_name: str) -> AutoTokenizer:
    '''Loads the corresponding tokenizer for a given model.'''
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer