from transformers import *
import torch

def main():
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
    model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
    input_text = "我觉得不太行！"
    inputs = tokenizer(input_text, return_tensors="pt")
    res = model(**inputs)
    logits = torch.softmax(res.logits, dim=-1)
    pred = torch.argmax(logits).item()
    result = model.config.id2label.get(pred)
    print(result, logits[0][pred].item())   


if __name__ == "__main__":
    main()
