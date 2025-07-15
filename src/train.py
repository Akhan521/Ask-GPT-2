from trl import SFTTrainer, SFTConfig
from src.data_loader import load_data, get_tokenizer
from src.model import load_model

def train(model_name: str, dataset_name: str) -> None:
    '''Trains the specified model on the given dataset.'''
    # Get our tokenizer and model.
    tokenizer = get_tokenizer(model_name)
    model = load_model(model_name)

    # Load the dataset.
    dataset = load_data(dataset_name)
    training_dataset = dataset['train']

    # Define training arguments.
    training_args = SFTConfig(
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        output_dir='logs',
        report_to='none',
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        group_by_length=True,
        max_length=512
    )

    # Create the SFT trainer.
    trainer = SFTTrainer(
        model=model,
        train_dataset=training_dataset,
        processing_class=tokenizer,
        args=training_args
    )

    # Start training.
    trainer.train()

    # Save the model and tokenizer.
    print("\nTraining complete. Saving model and tokenizer...")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    # Our model will be fine-tuned on the following dataset for Q&A.
    dataset_name = "mlabonne/guanaco-llama2-1k"

    # The model we will use.
    model_name = "distilgpt2"

    train(model_name, dataset_name)