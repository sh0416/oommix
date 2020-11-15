import os
import csv
import math
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import RobertaConfig, RobertaModel
from tqdm import tqdm

torch.set_printoptions(profile="full", linewidth=200)

#config = RobertaConfig.from_pretrained("roberta-base")
config = BertConfig.from_pretrained("bert-base-uncased")
print(config)

# AG News: https://www.kaggle.com/amananandrai/ag-news-classification-dataset

class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        print("Special token %s: %d" % (self.tokenizer.cls_token, self.tokenizer.cls_token_id))
        print("Special token %s: %d" % (self.tokenizer.sep_token, self.tokenizer.sep_token_id))

    def __call__(self, batch):
        inputs = self.tokenizer([x["input"] for x in batch], padding=True, max_length=256, return_tensors="pt")
        inputs["mixup_mask"] = inputs["attention_mask"] & \
                (inputs["input_ids"] != self.tokenizer.cls_token_id) & \
                (inputs["input_ids"] != self.tokenizer.sep_token_id)
        return {"inputs": inputs,
                "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long)}


def load_ag_news(filepath):
    with open(filepath, newline='', encoding="UTF8") as f:
        reader = csv.DictReader(f, fieldnames=["class", "title", "description"])
        data = []
        for row in reader:
            data.append({"label": int(row["class"]) - 1,
                        "input": row["title"]})
        print(any(['\\n' in row["input"] for row in data]))
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


def masked_argmax(vec, mask, dim, keepdim=False):
    vec_rank = torch.argsort(torch.argsort(vec, dim=dim), dim=dim) + 1
    masked_vec_rank = vec_rank * mask.float()
    return torch.argmax(masked_vec_rank, dim=dim, keepdim=keepdim)


class Bert(nn.Module):
    # For roberta model,
    #def __init__(self, vocab_size=50265, embed_dim=768, padding_idx=1, max_length=514, drop_prob=0.1,
    #             n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12):
    def __init__(self, vocab_size=30522, embed_dim=768, padding_idx=0, max_length=512, drop_prob=0.1,
                 n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_length, embed_dim, padding_idx=padding_idx)
        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        self.embedding_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.encoder = nn.ModuleList([
            nn.ModuleDict({
                "query": nn.Linear(embed_dim, n_head * k_dim),
                "key": nn.Linear(embed_dim, n_head * k_dim),
                "value": nn.Linear(embed_dim, n_head * v_dim),
                "out": nn.Linear(n_head * v_dim, embed_dim),
                "norm": nn.LayerNorm(embed_dim, eps=1e-12),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, feedforward_dim),
                    nn.GELU(),
                    nn.Linear(feedforward_dim, embed_dim),
                ),
                "ff_norm": nn.LayerNorm(embed_dim, eps=1e-12)
            })
            for _ in range(n_layer)
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_prob),
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Dropout(p=drop_prob),
            nn.Linear(128, 4)
        )
        self.dropout = nn.Dropout(p=drop_prob)
        self.padding_idx = padding_idx
        self.n_head = n_head
        self.k_dim = k_dim
        self.v_dim = v_dim

    def forward(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        h = self._forward_embedding(input_ids, batch, seq_len)
        for i, module_dict in enumerate(self.encoder):
            h = self._forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return self.classifier(torch.mean(h, dim=1))

    def _forward_embedding(self, input_ids, batch, seq_len):
        word = self.word_embeddings(input_ids)
        position_ids = torch.arange(0, seq_len, device=input_ids.device)
        position_ids = position_ids[None, :].expand(batch, -1)
        position = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(position_ids)
        token_type = self.token_type_embeddings(token_type_ids)
        h = self.dropout(self.embedding_norm(word + position + token_type))
        return h
        
    def _forward_layer(self, h, attention_mask, module_dict, batch, seq_len):
        q = module_dict["query"](h).view(batch, seq_len, self.n_head, self.k_dim)
        k = module_dict["key"](h).view(batch, seq_len, self.n_head, self.k_dim)
        v = module_dict["value"](h).view(batch, seq_len, self.n_head, self.v_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        a = torch.matmul(q, k) / math.sqrt(self.k_dim)
        a = masked_softmax(a, attention_mask[:, None, None, :], dim=3)
        o = torch.matmul(a, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous()
        o = o.view(batch, seq_len, -1)  # [b, h, s, d]
        h = module_dict["norm"](h + self.dropout(module_dict["out"](o)))
        h = module_dict["ff_norm"](h + self.dropout(module_dict["ff"](h)))
        return h

    def load(self):
        model = BertModel.from_pretrained("bert-base-uncased")
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


class TMixBert(Bert):
    #def __init__(self, vocab_size=50265, embed_dim=768, padding_idx=1, max_length=514, drop_prob=0.1,
    #             n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12, mixup_layer=0):
    def __init__(self, vocab_size=30522, embed_dim=768, padding_idx=0, max_length=512, drop_prob=0.1,
                 n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12, mixup_layer=0):
        super().__init__(vocab_size=vocab_size, embed_dim=embed_dim, padding_idx=padding_idx,
                         drop_prob=drop_prob, n_head=n_head, k_dim=k_dim, v_dim=v_dim,
                         feedforward_dim=feedforward_dim, n_layer=n_layer)
        self.mixup_layer = mixup_layer

    def forward(self, input_ids, attention_mask, mixup_indices=None, alpha=None):
        batch, seq_len = input_ids.shape
        with torch.no_grad():
            h = self._forward_embedding(input_ids, batch, seq_len)
        for i, module_dict in enumerate(self.encoder):
            if i == self.mixup_layer and mixup_indices is not None:
                h = alpha * h + (1 - alpha) * h[mixup_indices]
            h = self._forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return self.classifier(torch.mean(h, dim=1))


class PdistMixBert(TMixBert):
    def forward(self, input_ids, attention_mask, mixup_indices=None, mixup_mask=None, alpha=None):
        """
        :param mixup_indices: batch level shuffle index list
        :type mixup_indices: torch.LongTensor(B)
        :param mixup_mask: token level mask array. True for normal token False for special token such as pad, cls, sep
        :type mixup_mask: torch.LongTensor(B, L)
        """
        batch, seq_len = input_ids.shape
        h = self._forward_embedding(input_ids, batch, seq_len)
        for i, module_dict in enumerate(self.encoder):
            if i == self.mixup_layer and mixup_indices is not None:
                with torch.no_grad():
                    mixup_mask = torch.logical_and(mixup_mask[:, :, None], mixup_mask[mixup_indices][:, None, :]) # [B, L1, L2]
                    mixup_mask = torch.logical_or(mixup_mask, torch.eye(seq_len, device=mixup_mask.device)[None, :, :]) 
                    #print(mixup_indices)
                    #print(mixup_mask)
                    q = module_dict["query"](h).view(batch, seq_len, self.n_head, self.k_dim)
                    k = module_dict["key"](h).view(batch, seq_len, self.n_head, self.k_dim)
                    q = q.permute(0, 2, 1, 3)
                    k = k.permute(0, 2, 3, 1)
                    # Challenge. Query vs Key? Query vs Query? Key vs Key?
                    cross_a = torch.matmul(q, k[mixup_indices]) / math.sqrt(self.k_dim) # [B, H, L1, L2]
                    cross_a = masked_softmax(cross_a, attention_mask[mixup_indices][:, None, None, :], dim=3)
                    # Challenge. Multi-head similarity? mean? max? approx?
                    cross_sim = torch.max(cross_a, dim=1)[0] # [B, L1, L2]
                    mixup_position = masked_argmax(cross_sim, mixup_mask, dim=2, keepdim=True) # [B, L1, 1]
                    #print(mixup_indices)
                    #print(mixup_mask)
                    #print("POSITION")
                    #print(mixup_position[:, :, 0])
                h2 = torch.gather(input=h[mixup_indices], dim=1, index=mixup_position.expand(-1, -1, h.shape[2]))
                #h[mixup_mask.bool()] = alpha * h[mixup_mask.bool()] + (1 - alpha) * h2[mixup_mask.bool()]
                h = alpha * h + (1 - alpha) * h2
            h = self._forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return self.classifier(torch.mean(h, dim=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model hyperparameter
    parser.add_argument("--restore", type=str)
    # Train hyperparameter
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=96)
    # Train hyperparameter - augmentation
    parser.add_argument("--augment", type=str, choices=["none", "tmix", "pdistmix"], default="none")
    parser.add_argument("--mixup_layer", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    train_dataset = AGNewsDataset(os.path.join("dataset", "ag_news_csv", "train.csv"))
    test_dataset = AGNewsDataset(os.path.join("dataset", "ag_news_csv", "test.csv"))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.augment == "none":
        model = Bert()
    elif args.augment == "tmix":
        model = TMixBert(mixup_layer=args.mixup_layer)
    elif args.augment == "pdistmix":
        model = PdistMixBert(mixup_layer=args.mixup_layer)

    model.load()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CollateFn(tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=CollateFn(tokenizer))

    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.Adam(itertools.chain(model.word_embeddings.parameters(),
                                             model.position_embeddings.parameters(),
                                             model.token_type_embeddings.parameters(),
                                             model.embedding_norm.parameters(),
                                             model.encoder.parameters()), lr=1e-5),
                  optim.Adam(model.classifier.parameters(), lr=1e-3)]
    schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, 1))
                for optimizer in optimizers]

    print("CUDA: %d" % torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter()

    if args.restore is not None:
        checkpoint = torch.load(args.restore)
        model.load_state_dict(checkpoint["model"])
        for optimizer, restored_state_dict in zip(optimizers, checkpoints["optimizers"]):
            optimizer.load_state_dict(restored_state_dict)
        for scheduler, restored_state_dict in zip(schedulers, checkpoints["schedulers"]):
            scheduler.load_state_dict(restored_state_dict)

    writer.add_hparams(hparam_dict=vars(args), metric_dict={})
    step, best_acc = 0, 0
    model.to(device)
    for epoch in range(args.epoch):
        with tqdm(train_loader, desc="Epoch %d" % epoch, ncols=200) as tbar:
            for batch in tbar:
                input_ids = batch["inputs"]["input_ids"].to(device) 
                attention_mask = batch["inputs"]["attention_mask"].to(device) 
                if "mixup_mask" in batch["inputs"]:
                    mixup_mask = batch["inputs"]["mixup_mask"].to(device)
                else:
                    mixup_mask = None
                labels = batch["labels"].to(device)

                if args.augment == "none":
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    tbar.set_postfix(loss="%.4f" % loss.item())
                elif args.augment == "tmix" or args.augment == "pdistmix":
                    mixup_indices = torch.randperm(input_ids.shape[0], device=device)
                    lambda_ = np.random.beta(args.alpha, args.alpha)
                    if lambda_ < 0.5:
                        lambda_ = 1 - lambda_
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, mixup_mask=mixup_mask, alpha=lambda_)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(outputs, labels[mixup_indices])
                    loss = lambda_ * loss1 + (1 - lambda_) * loss2
                    tbar.set_postfix(loss1="%.4f" % loss1, loss2="%.4f" % loss2, loss="%.4f" % loss.item())
                writer.add_scalar("train_loss", loss, global_step=step)
                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss.backward()
                # Track gradient norm
                with torch.no_grad():
                    for k, v in model.named_parameters():
                        writer.add_scalar("grad/"+k, v.grad.data.norm(2), global_step=step)
                for optimizer in optimizers:
                    optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()
                if step % 500 == 0:
                    with torch.no_grad():
                        model.eval()
                        with tqdm(test_loader, desc="Evaluate test", ncols=200, leave=True, position=0) as test_tbar:
                            correct, count = 0, 0
                            for batch in test_tbar:
                                input_ids = batch["inputs"]["input_ids"].to(device) 
                                attention_mask = batch["inputs"]["attention_mask"].to(device) 
                                labels = batch["labels"].to(device)
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                                pred = outputs.argmax(dim=1)
                                correct += (labels == pred).float().sum()
                                count += labels.shape[0]
                                test_tbar.set_postfix(test_acc=(correct/count).item())
                            acc = correct / count
                            writer.add_scalar("test_acc", acc, global_step=step)
                        model.train()
                    if acc > best_acc:
                        torch.save({"epoch": epoch,
                                    "model": model.state_dict(),
                                    "optimizer": [optimizer.state_dict() for optimizer in optimizers],
                                    "scheduler": [scheduler.state_dict() for scheduler in schedulers]},
                                   "checkpoint_best.pt")
                step += 1

