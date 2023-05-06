import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def load_dataset(train_file, test_file, tokenizer):
    # Load the training and testing datasets as TextDataset objects
    # using the specified tokenizer and block_size
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
    test_dataset = TextDataset(tokenizer=tokenizer, file_path=test_file, block_size=128)
    return train_dataset, test_dataset

def main():
    # Set the model and tokenizer
    # model_name can be "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set training and testing dataset files
    train_file = 'path/to/your/train_data.txt'
    test_file = 'path/to/your/test_data.txt'
    # Remember to replace 'path/to/your/train_data.txt' and 'path/to/your/test_data.txt' 
    # with the actual file paths of your training and testing datasets before running the script.

    # Load datasets
    train_dataset, test_dataset = load_dataset(train_file, test_file, tokenizer)

if __name__ == '__main__':
    main()
