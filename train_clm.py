from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="EleutherAI/pythia-160m",
        help="Model name on HuggingFace",
    )
    parser.add_argument(
        "--data_path", required=True, help='Path to a .csv file with a "text" column'
    )
    parser.add_argument(
        "--train_test_ratio", default=0.9, help="train/test split ratio"
    )
    parser.add_argument("--batch_size", default=2, help="batch size")
    parser.add_argument("--epochs", default=3, help="# of epochs")
    parser.add_argument("--output_dir", default="outputs", help="output dir path")

    parser.add_argument("--deepspeed", default=None, help="deepspeed config")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    return args


def tokenize_fn(examples, tokenizer):
    tokenized = tokenizer(examples["text"], truncation=True)
    examples["input_ids"] = tokenized["input_ids"]
    examples["attention_mask"] = tokenized["attention_mask"]
    return examples


def get_dataset(data_path: str, train_test_ratio, tokenizer):
    # Load the csv
    dataset = load_dataset("csv", data_files=data_path)

    # Clear out faulty samples
    dataset = dataset.filter(lambda example: example['text'] is not None)

    # Tokenize "text" column
    dataset = dataset.map(tokenize_fn, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    # Drop all unneeded columns
    to_drop = [
        col
        for col in dataset.column_names["train"]
        if col not in ["input_ids", "attention_mask"]
    ]
    dataset = dataset.remove_columns(to_drop)

    # Split
    dataset = dataset["train"].train_test_split(train_size=train_test_ratio)
    return dataset


def main(args):
    # Build trainer args
    # Note: This MUST be before loading the model.
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#constructing-massive-models
    trainer_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=1,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Get the dataset dict
    dataset = get_dataset(args.data_path, args.train_test_ratio, tokenizer)

    # Load the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Init the trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        args=trainer_args,
    )
    # Train
    trainer.train()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
