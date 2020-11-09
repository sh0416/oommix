import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm


# AG News: https://www.kaggle.com/amananandrai/ag-news-classification-dataset

class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return {"inputs": self.tokenizer([x["input"] for x in batch], padding=True, return_tensors="pt"),
                "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long)}


def load_ag_news(filepath):
    with open(filepath, newline='', encoding="UTF8") as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            data.append({"label": int(row["Class Index"]) - 1,
                        "input": row["Title"]})
    return data


class AGNewsDataset(Dataset):
    def __init__(self, filepath):
        self.data = load_ag_news(filepath)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


train_dataset = AGNewsDataset(os.path.join("dataset", "ag_news", "train.csv"))
test_dataset = AGNewsDataset(os.path.join("dataset", "ag_news", "test.csv"))

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", return_dict=True, num_labels=4)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=CollateFn(tokenizer))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=CollateFn(tokenizer))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
with tqdm(train_loader) as tbar:
    for batch in tbar:
        batch["inputs"]["input_ids"] = batch["inputs"]["input_ids"].to(device) 
        batch["inputs"]["attention_mask"] = batch["inputs"]["attention_mask"].to(device) 
        batch["labels"] = batch["labels"].to(device)
        outputs = model(**batch["inputs"])
        loss = criterion(outputs["logits"], batch["labels"])
        tbar.set_postfix(loss=loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
