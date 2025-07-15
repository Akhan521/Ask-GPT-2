from src.data_loader import get_tokenizer
from src.model import load_model
from src.generate import generate
import torch

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine whether to use a fine-tuned model or the base model.
    use_base = False             # Set this to 'True' if you want to use the base model.
    use_local_finetuned = False  # Set this to 'True' if you want to load your locally fine-tuned model from the 'logs' directory.

    # The model we will use (My fine-tuned model is saved in the Hugging Face Hub).
    model_path = "distilgpt2" if use_base else "logs" if use_local_finetuned else "akhan365/distilgpt2-finetuned-on-guanaco-for-qa"

    # Load the tokenizer and model.
    print(f"\nLoading model and tokenizer from {model_path}...")
    tokenizer = get_tokenizer(model_path)
    model = load_model(model_path)

    # Play around with the prompt you want to generate text for.
    prompt = "how are you"

    print(f'\nGenerating text using {'fine-tuned' if not use_base else 'base'} model...')
    print('=' * 50)
    print(f'Prompt: "{prompt}"\n')
    generated_text = generate(prompt, model, tokenizer, device)
    print(f'\nGenerated text: "{generated_text}"\n')