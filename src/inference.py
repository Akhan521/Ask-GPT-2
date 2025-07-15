from src.data_loader import load_data, get_tokenizer
from src.model import load_model
from src.generate import generate
import torch

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The model we will use.
    model_name = "distilgpt2"

    # Load the tokenizer and model.
    tokenizer = get_tokenizer(model_name)
    model = load_model(model_name)

    prompt = "how are you?"

    print()
    generated_text = generate(prompt, model, tokenizer, device)
    print()