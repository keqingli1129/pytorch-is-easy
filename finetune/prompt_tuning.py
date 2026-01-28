from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import load_dataset

def process_func(example, tokenizer):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    ds = load_dataset("shibing624/alpaca-zh")
    # dataset.save_to_disk("data/alpaca_zh/")
    # ds = Dataset.load_from_disk("data/alpaca_zh/")
    # Add further processing or training logic here
    # print(ds["train"][:3])
    tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
    tokenized_ds = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir="./chatbot",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=1
    )
    model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True)
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    # model = model.cuda()
    # ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
    # tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()