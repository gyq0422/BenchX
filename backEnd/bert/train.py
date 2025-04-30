import sys
import os
import gradio as gr
from io import StringIO
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score

# 定义一个类来捕获标准输出
class ConsoleOutput:
    def __init__(self):
        self.output = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.output  # 重定向标准输出

    def get_output(self):
        return self.output.getvalue()

    def reset(self):
        sys.stdout = self.old_stdout  # 恢复标准输出

def evaluate_model(dataset_name: str, output_dir: str = "./results", local_data_dir: str = "./local_data"):
    """
    Evaluate a pre-trained BERT model for a given dataset, using local cache if available.
    This function will yield progress updates during execution.
    """
    console_output = ConsoleOutput()  # 创建 ConsoleOutput 实例，捕捉控制台输出
    try:
        yield "Loading dataset..."
        dataset_params = {
            "ag_news": {"num_labels": 4, "text_field": "text"},
            "imdb": {"num_labels": 2, "text_field": "text"},
            "yelp_polarity": {"num_labels": 2, "text_field": "text"},
        }

        if dataset_name not in dataset_params:
            yield f"Unsupported dataset: '{dataset_name}'. Supported: {list(dataset_params.keys())}"
            return None

        num_labels = dataset_params[dataset_name]["num_labels"]
        text_field = dataset_params[dataset_name]["text_field"]

        # Create local data path
        local_dataset_path = os.path.join(local_data_dir, dataset_name)
        local_tokenizer_path = os.path.join(local_data_dir, "bert-base-uncased-tokenizer")

        # Ensure local directories exist
        os.makedirs(local_data_dir, exist_ok=True)

        # 1. Load dataset (from cache if available)
        if os.path.exists(local_dataset_path):
            yield f"Loading dataset from local path: {local_dataset_path}"
            dataset = load_from_disk(local_dataset_path)
        else:
            yield f"Downloading dataset: {dataset_name}"
            try:
                dataset = load_dataset(dataset_name)
                yield f"Saving dataset to local path: {local_dataset_path}"
                dataset.save_to_disk(local_dataset_path)
            except Exception as e:
                yield f"Error loading dataset '{dataset_name}': {e}"
                return None

        # 2. Load tokenizer (from cache if available)
        if os.path.exists(local_tokenizer_path):
            yield f"Loading tokenizer from local path: {local_tokenizer_path}"
            tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path)
        else:
            yield "Downloading tokenizer: bert-base-uncased"
            try:
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                yield f"Saving tokenizer to local path: {local_tokenizer_path}"
                tokenizer.save_pretrained(local_tokenizer_path)
            except Exception as e:
                yield f"Error loading tokenizer: {e}"
                return None

        # Tokenize function
        def tokenize_function(examples):
            return tokenizer(examples[text_field], padding="max_length", truncation=True)

        # Check for 'test' or 'validation' split
        if 'test' not in dataset:
            eval_split_name = 'validation' if 'validation' in dataset else None
            if eval_split_name is None:
                yield f"Error: Dataset '{dataset_name}' has no 'test' or 'validation' split for evaluation."
                return None
        else:
            eval_split_name = 'test'

        try:
            yield f"Tokenizing evaluation split: {eval_split_name}"
            eval_dataset = dataset[eval_split_name].map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.remove_columns([text_field])
            eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        except Exception as e:
            yield f"Error processing dataset: {e}"
            return None

        # Load the model
        try:
            yield "Loading pre-trained model: bert-base-uncased"
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        except Exception as e:
            yield f"Error loading model: {e}"
            return None

        # Evaluation setup
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, dataset_name + "_eval"),
            per_device_eval_batch_size=16,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions.argmax(axis=-1), p.label_ids)},
            tokenizer=tokenizer
        )

        # Start evaluation
        try:
            yield "Starting evaluation..."
            results = trainer.evaluate()
            yield f"Evaluation finished. Results for {dataset_name}: {results}"
            return results
        except Exception as e:
            yield f"Error during evaluation for {dataset_name}: {e}"
            return None
    finally:
        # 在最终步骤中，捕获所有的控制台输出并显示在Gradio前端
        output = console_output.get_output()
        if output:
            yield f"Captured Output:\n{output}"
        console_output.reset()

# --- 主执行块：依次评估指定的数据集 --- 
if __name__ == "__main__":
    datasets_to_evaluate = ["ag_news", "imdb", "yelp_polarity"]
    base_output_dir = "./results"
    local_data_dir = "./local_cache"
    all_results = {}

    for ds_name in datasets_to_evaluate:
        # num_labels 会在 evaluate_model 内部根据 ds_name 确定
        eval_results = evaluate_model(
            dataset_name=ds_name, 
            output_dir=base_output_dir, 
            local_data_dir=local_data_dir
        )
        if eval_results:
            all_results[ds_name] = eval_results
        else:
            print(f"Skipping results for {ds_name} due to errors.")

    print("\n--- All Evaluation Results ---")
    for ds_name, res in all_results.items():
        print(f"Dataset: {ds_name}, Accuracy: {res.get('eval_accuracy', 'N/A'):.4f}")
