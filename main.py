import os
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import RobertaConfig, RobertaModel
from tqdm import tqdm

config = RobertaConfig.from_pretrained("roberta-base")
print(config)

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


def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums


class Roberta(nn.Module):
    def __init__(self, vocab_size=50265, embed_dim=768, padding_idx=1, max_length=514, drop_prob=0.1,
                 n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_length, embed_dim, padding_idx=padding_idx)
        self.token_type_embeddings = nn.Embedding(1, embed_dim)
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.ModuleList([
            nn.ModuleDict({
                "query": nn.Linear(embed_dim, n_head * k_dim),
                "key": nn.Linear(embed_dim, n_head * k_dim),
                "value": nn.Linear(embed_dim, n_head * v_dim),
                "out": nn.Linear(n_head * v_dim, embed_dim),
                "norm": nn.LayerNorm(embed_dim),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, feedforward_dim),
                    nn.GELU(),
                    nn.Linear(feedforward_dim, embed_dim),
                ),
                "ff_norm": nn.LayerNorm(embed_dim)
            })
            for _ in range(n_layer)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 4)
        )
        self.dropout = nn.Dropout(p=drop_prob)
        self.padding_idx = padding_idx
        self.n_head = n_head
        self.k_dim = k_dim
        self.v_dim = v_dim

    def forward(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        word = self.word_embeddings(input_ids)
        position_ids = torch.arange(self.padding_idx + 1, self.padding_idx + 1 + seq_len,
                                    device=input_ids.device)
        position_ids = position_ids[None, :].expand(batch, -1)
        position_ids = torch.where(attention_mask.bool(), position_ids, torch.ones_like(position_ids))
        position = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(position_ids)
        token_type = self.token_type_embeddings(token_type_ids)
        h = self.dropout(self.embedding_norm(word + position + token_type))
        for i, module_dict in enumerate(self.encoder):
            q = module_dict["query"](h).view(batch, seq_len, self.n_head, self.k_dim)
            k = module_dict["key"](h).view(batch, seq_len, self.n_head, self.k_dim)
            v = module_dict["value"](h).view(batch, seq_len, self.n_head, self.v_dim)
            a = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) / math.sqrt(self.k_dim)
            a = masked_softmax(a, attention_mask[:, None, None, :], dim=3)
            o = torch.matmul(a, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous()
            o = o.view(batch, seq_len, -1)  # [b, h, s, d]
            h = module_dict["norm"](h + self.dropout(module_dict["out"](o)))
            h = module_dict["ff_norm"](h + self.dropout(module_dict["ff"](h)))
        return h

    def load(self):
        model = RobertaModel.from_pretrained("roberta-base")
        self.word_embeddings.load_state_dict(model.embeddings.word_embeddings.state_dict())
        self.position_embeddings.load_state_dict(model.embeddings.position_embeddings.state_dict())
        self.token_type_embeddings.load_state_dict(model.embeddings.token_type_embeddings.state_dict())
        self.embedding_norm.load_state_dict(model.embeddings.LayerNorm.state_dict())
        for t, f in zip(self.encoder, model.encoder.layer):
            t["query"].load_state_dict(f.attention.self.query.state_dict())
            t["key"].load_state_dict(f.attention.self.key.state_dict())
            t["value"].load_state_dict(f.attention.self.value.state_dict())
            t["out"].load_state_dict(f.attention.output.dense.state_dict())
            t["norm"].load_state_dict(f.attention.output.LayerNorm.state_dict())
            t["ff"][0].load_state_dict(f.intermediate.dense.state_dict())
            t["ff"][2].load_state_dict(f.output.dense.state_dict())
            t["ff_norm"].load_state_dict(f.output.LayerNorm.state_dict())


train_dataset = AGNewsDataset(os.path.join("dataset", "ag_news", "train.csv"))
test_dataset = AGNewsDataset(os.path.join("dataset", "ag_news", "test.csv"))


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", output_hidden_states=True,
output_attentions=True, return_dict=True, num_labels=4)
print(model)
my_model = Roberta()
my_model.load()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=CollateFn(tokenizer))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=CollateFn(tokenizer))

criterion = nn.CrossEntropyLoss()
optimizers = [optim.Adam(model.roberta.parameters(), lr=1e-5),
              optim.Adam(model.classifier.parameters(), lr=1e-3)]
schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, 1))
              for optimizer in optimizers]

print("CUDA: %d" % torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()
step, best_acc = 0, 0
model.to(device)
for epoch in range(10):
    with tqdm(train_loader, desc="Epoch %d" % epoch) as tbar:
        for batch in tbar:
            model.eval()
            my_model.eval()
            batch["inputs"]["input_ids"] = batch["inputs"]["input_ids"].to(device) 
            batch["inputs"]["attention_mask"] = batch["inputs"]["attention_mask"].to(device) 
            batch["labels"] = batch["labels"].to(device)
            outputs = model(**batch["inputs"])
            my_outputs = my_model(**batch["inputs"])
            #print((my_outputs[0] - outputs["attentions"][0]).abs().max())
            #print((my_outputs[1] - outputs["hidden_states"][1]).abs().max())
            print((my_outputs - outputs["hidden_states"][12]).abs().max())
            assert False
            loss = criterion(outputs["logits"], batch["labels"])
            tbar.set_postfix(loss=loss.item())
            writer.add_scalar("train_loss", loss, global_step=step)
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            for scheduler in schedulers:
                scheduler.step()
            if step % 500 == 0:
                with torch.no_grad():
                    model.eval()
                    with tqdm(test_loader, desc="Evaluate test") as test_tbar:
                        correct, count = 0, 0
                        for batch in test_tbar:
                            batch["inputs"]["input_ids"] = batch["inputs"]["input_ids"].to(device) 
                            batch["inputs"]["attention_mask"] = batch["inputs"]["attention_mask"].to(device) 
                            batch["labels"] = batch["labels"].to(device)
                            outputs = model(**batch["inputs"])
                            pred = outputs["logits"].argmax(dim=1)
                            correct += (batch["labels"] == pred).float().sum()
                            count += batch["labels"].shape[0]
                            test_tbar.set_postfix(test_acc=(correct/count).item())
                        acc = correct / count
                        writer.add_scalar("test_acc", acc, global_step=step)
                    model.train()
                if acc > best_acc:
                    torch.save(model.state_dict(), "model_best.pt")
            step += 1

