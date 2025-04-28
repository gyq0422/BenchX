import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def generate_text_with_gpt2(dataset_name: str, dataset2, output_dir: str = "./results", max_length: int = 50):
    """
    Fine-tune GPT-2 for text generation on a given dataset.

    Args:
    - dataset_name (str): The name of the dataset to be used (e.g., "wikitext").
    - output_dir (str): Directory to save results and model checkpoints.
    - max_length (int): The maximum length of the generated text.

    Returns:
    - generated_text (str): A sample of generated text.
    """

    # 1. Load the dataset (Here we are using "wikitext" as an example)
    dataset = load_dataset(dataset_name, dataset2)

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 设置 padding token，使用 eos_token 作为 pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # 现在可以使用 tokenizer 进行分词和 padding
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    # 接下来，您可以使用这个 tokenizer 进行数据集的映射处理
    train_dataset = dataset['train'].map(tokenize_function, batched=True)
    test_dataset = dataset['test'].map(tokenize_function, batched=True)

    # 5. Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,               # Output directory for model checkpoints
        num_train_epochs=3,                  # Number of epochs
        per_device_train_batch_size=4,       # Batch size for training
        per_device_eval_batch_size=4,        # Batch size for evaluation
        save_steps=500,                       # Save model every 500 steps
        save_total_limit=2,                  # Keep only the last 2 saved models
        logging_dir="./logs",                # Logging directory
    )

    # 6. Define Trainer
    trainer = Trainer(
        model=model,                         # GPT-2 model
        args=training_args,                  # Training arguments
        train_dataset=train_dataset,         # Training dataset
        eval_dataset=test_dataset,           # Evaluation dataset
    )

    # 7. Fine-tune the model
    trainer.train()

    # 8. Save the model and tokenizer
    model.save_pretrained(f"{output_dir}/saved_model")
    tokenizer.save_pretrained(f"{output_dir}/saved_model")

    # 9. Generate text with the fine-tuned model
    input_text = "Once upon a time"  # You can change the prompt
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text using GPT-2 model
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")

    return generated_text

# Example of how to call the function:
if __name__ == "__main__":
    dataset_name = "wikitext"  # You can change this to any other text dataset
    d = "wikitext-103-raw-v1"
    output_dir = "./gpt2_results"  # Directory to save the model and results

    generated_text = generate_text_with_gpt2(dataset_name,d, output_dir)
