import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score


def train_and_evaluate(dataset_name: str, num_labels: int, output_dir: str = "./results"):
    """
    Train and evaluate a BERT model for a given dataset.

    Args:
    - dataset_name (str): The name of the dataset to be used (e.g., "ag_news").
    - num_labels (int): The number of classes in the dataset (e.g., 4 for "ag_news").
    - output_dir (str): Directory to save results and model checkpoints.

    Returns:
    - results (dict): The evaluation results.yu8
    """

    # 1. Load the dataset
    dataset = load_dataset(dataset_name)

    # 2. Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = dataset['train'].map(tokenize_function, batched=True)
    test_dataset = dataset['test'].map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,  # Output directory
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=8,  # Train batch size per device
        per_device_eval_batch_size=8,  # Eval batch size per device
        warmup_steps=500,  # Warmup steps
        weight_decay=0.01,  # Weight decay
        logging_dir="./logs",  # Logging directory
        logging_steps=10,  # Log every 10 steps
        save_steps=500,  # Save checkpoint every 500 steps
        save_total_limit=2,  # Keep the last 2 saved checkpoints
    )

    trainer = Trainer(
        model=model,  # Model
        args=training_args,  # Training arguments
        train_dataset=train_dataset,  # Train dataset
        eval_dataset=test_dataset,  # Eval dataset
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions.argmax(axis=-1), p.label_ids)},
        # Accuracy metric
    )

    trainer.train()

    results = trainer.evaluate()
    print(results)




# Example of how to call the function:
if __name__ == "__main__":
    dataset_name = "ag_news"  # You can change this to any other dataset
    num_labels = 4  # AG News has 4 labels, adjust as needed for your dataset
    output_dir = "./results"  # Folder to save the results and model

    results = train_and_evaluate(dataset_name, num_labels, output_dir)
