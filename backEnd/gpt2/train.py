import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
import os # 新增导入

def generate_text_with_gpt2(dataset_name: str, dataset_config_name: str, output_dir: str = "./results", local_data_dir: str = "./local_gpt2_cache", max_length: int = 50):
    """
    Fine-tune GPT-2 for text generation on a given dataset, with local caching.

    Args:
    - dataset_name (str): The name of the dataset to be used (e.g., "wikitext").
    - dataset_config_name (str): The specific configuration name for the dataset (e.g., "wikitext-103-raw-v1").
    - output_dir (str): Directory to save results and model checkpoints.
    - local_data_dir (str): Directory to cache/load the dataset and tokenizer.
    - max_length (int): The maximum length of the generated text.

    Returns:
    - generated_text (str): A sample of generated text.
    """

    # --- 本地缓存逻辑 --- 
    model_name = "gpt2"
    # 修正：为数据集和分词器创建独立的子目录
    local_dataset_path = os.path.join(local_data_dir, dataset_name, dataset_config_name)
    local_tokenizer_path = os.path.join(local_data_dir, model_name + "-tokenizer")

    # 确保缓存目录存在
    os.makedirs(local_dataset_path, exist_ok=True)
    os.makedirs(local_tokenizer_path, exist_ok=True)

    try:
        # 尝试从本地加载数据集
        print(f"Attempting to load dataset from local cache: {local_dataset_path}")
        # 检查数据集特征文件是否存在，作为判断缓存是否完整的依据
        if os.path.exists(os.path.join(local_dataset_path, 'dataset_info.json')) or os.path.exists(os.path.join(local_dataset_path, 'state.json')):
             dataset = load_from_disk(local_dataset_path)
             print("Dataset loaded successfully from local cache.")
        else:
             print("Local dataset cache incomplete or not found. Downloading...")
             dataset = load_dataset(dataset_name, dataset_config_name)
             print("Dataset downloaded. Saving to local cache...")
             dataset.save_to_disk(local_dataset_path)
             print(f"Dataset saved to {local_dataset_path}")

    except Exception as e:
        print(f"Failed to load dataset from cache or download failed: {e}. Attempting download...")
        dataset = load_dataset(dataset_name, dataset_config_name)
        print("Dataset downloaded. Saving to local cache...")
        dataset.save_to_disk(local_dataset_path)
        print(f"Dataset saved to {local_dataset_path}")

    try:
        # 尝试从本地加载分词器
        print(f"Attempting to load tokenizer from local cache: {local_tokenizer_path}")
        # 检查 tokenizer 配置文件是否存在
        if os.path.exists(os.path.join(local_tokenizer_path, 'tokenizer_config.json')):
            tokenizer = GPT2Tokenizer.from_pretrained(local_tokenizer_path)
            print("Tokenizer loaded successfully from local cache.")
        else:
            print("Local tokenizer cache not found. Downloading...")
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            print("Tokenizer downloaded. Saving to local cache...")
            tokenizer.save_pretrained(local_tokenizer_path)
            print(f"Tokenizer saved to {local_tokenizer_path}")

    except Exception as e:
        print(f"Failed to load tokenizer from cache or download failed: {e}. Attempting download...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("Tokenizer downloaded. Saving to local cache...")
        tokenizer.save_pretrained(local_tokenizer_path)
        print(f"Tokenizer saved to {local_tokenizer_path}")
    # --- 缓存逻辑结束 ---

    # 加载预训练模型 (通常不需要缓存，因为它可能很大且不常变动，或者由 transformers 库自行管理缓存)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 设置 padding token，使用 eos_token 作为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 更新模型的 pad_token_id，如果需要的话
        model.config.pad_token_id = tokenizer.eos_token_id

    # 现在可以使用 tokenizer 进行分词和 padding
    def tokenize_function(examples):
        # 确保使用正确的文本字段名，wikitext 通常是 'text'
        text_field = 'text'
        if text_field not in examples:
             potential_fields = [k for k, v in examples.items() if isinstance(v, list) and isinstance(v[0], str)]
             if potential_fields:
                 text_field = potential_fields[0]
             else:
                 raise ValueError(f"Cannot find text field in dataset features: {list(examples.keys())}")
        
        # Tokenize the text
        tokenized_output = tokenizer(examples[text_field], padding="max_length", truncation=True, max_length=128)
        
        # For language modeling, labels are usually the input_ids themselves.
        # The model internally shifts labels for loss calculation.
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        
        return tokenized_output

    # 接下来，您可以使用这个 tokenizer 进行数据集的映射处理
    # 检查数据集是否包含 'train' 和 'test' 拆分
    if 'train' not in dataset:
        raise ValueError(f"Dataset {dataset_name}/{dataset_config_name} does not contain a 'train' split.")
    if 'test' not in dataset:
        # 如果没有 'test' split，可以考虑使用 'validation' split 或者从 'train' split 中划分
        if 'validation' in dataset:
            print("Warning: 'test' split not found. Using 'validation' split for evaluation.")
            test_split_name = 'validation'
        else:
            # 如果连 'validation' 都没有，需要用户决定如何处理，这里暂时抛出错误
            raise ValueError(f"Dataset {dataset_name}/{dataset_config_name} does not contain a 'test' or 'validation' split for evaluation.")
    else:
        test_split_name = 'test'

    print("Tokenizing datasets...")
    # Remove `remove_columns` to let Trainer handle column selection based on model signature
    tokenized_train_dataset = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test_dataset = dataset[test_split_name].map(tokenize_function, batched=True)
    print("Tokenization complete.")

    # 5. Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,               # Output directory for model checkpoints
        num_train_epochs=1,                  # Number of epochs (can increase later)
        per_device_train_batch_size=2,       # Batch size for training (adjust based on memory)
        per_device_eval_batch_size=2,        # Batch size for evaluation
        gradient_accumulation_steps=2,       # Simulate larger batch size
        save_steps=1000,                     # Save checkpoint frequency
        save_total_limit=2,                  # Limit total checkpoints
        logging_dir=f"{output_dir}/logs",   # Logging directory
        logging_steps=100,                   # Log frequency
        fp16=torch.cuda.is_available(),      # Use FP16 if GPU supports it
    )

    # 6. Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
    )

    # 7. Fine-tune the model
    print("Starting model fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # 8. Save the final model and tokenizer
    final_save_path = os.path.join(output_dir, "final_model")
    print(f"Saving final model and tokenizer to {final_save_path}")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("Model and tokenizer saved.")

    # 9. Generate text with the fine-tuned model
    print("Generating text with the fine-tuned model...")
    input_text = "Once upon a time"  # You can change the prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device) # Ensure tensors are on the same device as the model

    # Generate text using GPT-2 model
    # 使用 beam search 或其他生成策略可能效果更好
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        num_beams=5, # Example: using beam search
        early_stopping=True
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")

    # 评估模型
    print("Evaluating the final model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    try:
        perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
        print(f"Perplexity: {perplexity.item()}")
    except KeyError:
        print("Could not calculate perplexity from evaluation results.")

    return generated_text, eval_results

# Example of how to call the function:
if __name__ == "__main__":
    dataset_name = "wikitext"  # Dataset name on Hugging Face Hub
    dataset_config_name = "wikitext-2-raw-v1" # Specific configuration (use a smaller one for faster example)
    output_dir = os.path.join(".", "gpt2_finetuned_results") # Directory to save the model and results
    local_cache_dir = os.path.join(".", "local_gpt2_cache") # Directory for local cache

    print(f"Starting GPT-2 fine-tuning for dataset: {dataset_name}/{dataset_config_name}")
    print(f"Results will be saved to: {output_dir}")
    print(f"Local cache directory: {local_cache_dir}")

    generated_text, eval_metrics = generate_text_with_gpt2(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        output_dir=output_dir,
        local_data_dir=local_cache_dir,
        max_length=100 # Generate longer text
    )

    print("\n--- Script Finished ---")
    print(f"Final Generated Text Sample: {generated_text}")
    print(f"Final Evaluation Metrics: {eval_metrics}")
