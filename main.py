import os
import csv
import math
import pprint
import argparse
import itertools
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertConfig, BertTokenizerFast
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import RobertaConfig, RobertaModel
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_printoptions(profile="full", linewidth=150)

#config = RobertaConfig.from_pretrained("roberta-base")
config = BertConfig.from_pretrained("bert-base-uncased")
print(config)

# AG News: https://www.kaggle.com/amananandrai/ag-news-classification-dataset

class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        print("Special token %s: %d" % (self.tokenizer.cls_token, self.tokenizer.cls_token_id))
        print("Special token %s: %d" % (self.tokenizer.sep_token, self.tokenizer.sep_token_id))
        print("Special token %s: %d" % (self.tokenizer.pad_token, self.tokenizer.pad_token_id))

    def __call__(self, batch):
        inputs = {}
        with torch.no_grad():
            inputs["input_ids"] = nn.utils.rnn.pad_sequence([x["input"] for x in batch],
                                                            batch_first=True,
                                                            padding_value=self.tokenizer.pad_token_id)
            inputs["attention_mask"] = inputs["input_ids"] != self.tokenizer.pad_token_id
            inputs["mixup_mask"] = {
                    "is_cls": (inputs["input_ids"] == self.tokenizer.cls_token_id),
                    "is_sep": (inputs["input_ids"] == self.tokenizer.sep_token_id),
                    "is_normal": ((inputs["input_ids"] != self.tokenizer.cls_token_id) & \
                                  (inputs["input_ids"] != self.tokenizer.sep_token_id) & \
                                  (inputs["input_ids"] != self.tokenizer.pad_token_id))}
            labels = torch.stack([x["label"] for x in batch])
        return {"inputs": inputs, "labels": labels}


class ListDataset(Dataset):
    def __getitem__(self, idx):
        data = self.data[idx]
        return {"input": torch.tensor(data["input"], dtype=torch.long),
                "label": torch.tensor(data["label"], dtype=torch.long)}
    
    def __len__(self):
        return len(self.data)


def load_csv(filepath, fieldnames=None):
    with open(filepath, newline='', encoding="UTF8") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            yield row


def save_csv(filepath, data, fieldnames): 
    with open(filepath, 'w', newline='', encoding='UTF8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def preprocess(f, filepath, tokenizer):
    cached_filepath = os.path.join('cache', filepath)
    if not os.path.exists(cached_filepath):
        data = f(filepath)
        for row in tqdm(data, desc="Tokenize amazon text"):
            row["input"] = ' '.join(map(str, tokenizer(row["input"], max_length=512, truncation=True)["input_ids"]))
        os.makedirs(os.path.dirname(cached_filepath), exist_ok=True)
        save_csv(cached_filepath, data, ["input", "label"])
    data = list(load_csv(cached_filepath))
    data = [{"input": list(map(int, row["input"].split(' '))), "label": int(row["label"])}
            for row in data]
    return data


def load_ag_news(filepath):
    data = [{"label": int(row["class"]) - 1, "input": row["title"]}
            for row in tqdm(load_csv(filepath, ["class", "title", "description"]), desc="Load ag news dataset")]
    return data


class AGNewsDataset(ListDataset):
    def __init__(self, filepath, tokenizer, data_size):
        self.data = preprocess(load_ag_news, filepath, tokenizer)
        self.n_class = 4
        if data_size != -1:
            self.data = self.data[:min(data_size, len(self.data))]


def load_amazon_review_full(filepath):
    data = [{"label": int(row["class"]) - 1, "input": row["text"]}
            for row in tqdm(load_csv(filepath, ["class", "title", "text"]), desc="Load amazon dataset")]
    data = data[:20000]
    return data


class AmazonReviewFullDataset(ListDataset):
    def __init__(self, filepath, tokenizer):
        self.data = preprocess(load_amazon_review_full, filepath, tokenizer)
        self.n_class = 5


def load_yelp_polarity(filepath):
    data = [{"label": int(row["class"]) - 1, "input": row["review"]}
            for row in tqdm(load_csv(filepath, ["class", "review"]), desc="Load yelp polarity")]
    return data


class YelpPolarityDataset(ListDataset):
    def __init__(self, filepath, tokenizer):
        self.data = preprocess(load_yelp_polarity, filepath, tokenizer)
        self.n_class = 2


def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums


def masked_argmax(vec, mask, dim, keepdim=False):
    vec_rank = torch.argsort(torch.argsort(vec, dim=dim), dim=dim) + 1
    masked_vec_rank = vec_rank * mask.float()
    return torch.argmax(masked_vec_rank, dim=dim, keepdim=keepdim)


class Bert(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=768, padding_idx=0, max_length=512, drop_prob=0.1,
                 n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12, n_class=4):
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
        self.dropout = nn.Dropout(p=drop_prob)
        self.padding_idx = padding_idx
        self.n_head = n_head
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        h = self.forward_embedding(input_ids, batch, seq_len)
        for i, module_dict in enumerate(self.encoder):
            h = self.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return h

    def forward_embedding(self, input_ids, batch, seq_len):
        word = self.word_embeddings(input_ids)
        position_ids = torch.arange(0, seq_len, device=input_ids.device)
        position_ids = position_ids[None, :].expand(batch, -1)
        position = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(position_ids)
        token_type = self.token_type_embeddings(token_type_ids)
        h = self.dropout(self.embedding_norm(word + position + token_type))
        return h
        
    def forward_layer(self, h, attention_mask, module_dict, batch, seq_len):
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


class TMix(nn.Module):
    def __init__(self, embedding_model, mixup_layer=0):
        super().__init__()
        self.embedding_model = embedding_model
        self.mixup_layer = mixup_layer

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        batch, seq_len = input_ids.shape
        h = self.embedding_model.forward_embedding(input_ids, batch, seq_len)
        for module_dict in self.embedding_model.encoder[:self.mixup_layer]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        if mixup_indices is not None:
            h = lambda_ * h + (1 - lambda_) * h[mixup_indices]
        for module_dict in self.embedding_model.encoder[self.mixup_layer:]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return h


class AdaMix(nn.Module):
    def __init__(self, embedding_model, mixup_layer=0):
        super().__init__()
        self.embedding_model = embedding_model
        self.policy_region_generator = nn.Sequential(
            nn.Linear(2*self.embedding_model.embed_dim, 3),
            nn.Softmax(dim=1))
        self.intrusion_classifier = nn.Linear(self.embedding_model.embed_dim, 1)
        self.mixup_layer = mixup_layer

    def forward(self, input_ids, attention_mask, mixup_indices=None):
        batch, seq_len = input_ids.shape
        h = self.embedding_model.forward_embedding(input_ids, batch, seq_len)

        for module_dict in self.embedding_model.encoder[:self.mixup_layer]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)

        if mixup_indices is not None:
            # Policy Region Generator
            sentence_h = h.mean(dim=1)
            policy_region = self.policy_region_generator(torch.cat((sentence_h, sentence_h[mixup_indices]), dim=1))  # [B, 3]
            eps = torch.rand(policy_region.shape[0], device=policy_region.device)
            gamma = policy_region[:, 1] * eps + policy_region[:, 0]
            mix_h = gamma[:, None, None] * h + (1 - gamma)[:, None, None] * h[mixup_indices]

        for module_dict in self.embedding_model.encoder[self.mixup_layer:]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
            if mixup_indices is not None:
                mix_h = self.embedding_model.forward_layer(mix_h, attention_mask, module_dict, batch, seq_len)

        # Classifier
        if mixup_indices is None:
            return h
        else:
            # Intrusion Discriminator
            intr = self.intrusion_classifier(h)
            mix_intr = self.intrusion_classifier(mix_h)
            return h, mix_h, intr, mix_intr, gamma

    def predict(self, input_ids, attention_mask):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask)


class SentenceClassificationModel(nn.Module):
    def __init__(self, embedding_model, n_class):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_model.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, n_class)
        )

    def forward(self, input_ids, attention_mask):
        h = self.embedding_model(input_ids, attention_mask)
        return self.classifier(torch.mean(h, dim=1))

    def load(self):
        self.embedding_model.load()


class TMixSentenceClassificationModel(nn.Module):
    def __init__(self, mix_model, n_class):
        super().__init__()
        self.mix_model = mix_model
        self.classifier = nn.Sequential(
            nn.Linear(self.mix_model.embedding_model.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, n_class)
        )

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        h = self.mix_model(input_ids, attention_mask, mixup_indices=mixup_indices, lambda_=lambda_)
        return self.classifier(torch.mean(h, dim=1))

    def load(self):
        self.mix_model.embedding_model.load()


class AdaMixSentenceClassificationModel(nn.Module):
    def __init__(self, mix_model, n_class):
        super().__init__()
        self.mix_model = mix_model
        self.classifier = nn.Sequential(
            nn.Linear(self.mix_model.embedding_model.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, n_class)
        )

    def forward(self, input_ids, attention_mask, mixup_indices=None):
        if mixup_indices is None:
            h = self.mix_model(input_ids, attention_mask)
            return self.classifier(torch.mean(h, dim=1))
        else:
            h, mix_h, intr, mix_intr, gamma = self.mix_model(input_ids, attention_mask, mixup_indices)
            out = self.classifier(torch.mean(h, dim=1))
            mix_out = self.classifier(torch.mean(mix_h, dim=1))
            return out, mix_out, intr, mix_intr, gamma
        
    def load(self):
        self.mix_model.embedding_model.load()


def create_bert_sentence_classification_model(vocab_size=30522, embed_dim=768, padding_idx=0,
        drop_prob=0.1, n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12, n_class=4):
    embedding_model = Bert(vocab_size=vocab_size, embed_dim=embed_dim, padding_idx=padding_idx,
                           drop_prob=drop_prob, n_head=n_head, k_dim=k_dim, v_dim=v_dim,
                           feedforward_dim=feedforward_dim, n_layer=n_layer)
    return SentenceClassificationModel(embedding_model, n_class)


def create_tmix_bert_sentence_classification_model(vocab_size=30522, embed_dim=768, padding_idx=0,
        drop_prob=0.1, n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12, mixup_layer=3, n_class=4):
    embedding_model = Bert(vocab_size=vocab_size, embed_dim=embed_dim, padding_idx=padding_idx,
                           drop_prob=drop_prob, n_head=n_head, k_dim=k_dim, v_dim=v_dim,
                           feedforward_dim=feedforward_dim, n_layer=n_layer)
    embedding_model = TMix(embedding_model, mixup_layer=mixup_layer)
    return TMixSentenceClassificationModel(embedding_model, n_class)


def create_adamix_bert_sentence_classification_model(vocab_size=30522, embed_dim=768, padding_idx=0,
        drop_prob=0.1, n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12, mixup_layer=3, n_class=4):
    embedding_model = Bert(vocab_size=vocab_size, embed_dim=embed_dim, padding_idx=padding_idx,
                           drop_prob=drop_prob, n_head=n_head, k_dim=k_dim, v_dim=v_dim,
                           feedforward_dim=feedforward_dim, n_layer=n_layer)
    embedding_model = AdaMix(embedding_model, mixup_layer=mixup_layer)
    return AdaMixSentenceClassificationModel(embedding_model, n_class)


class PdistMixBert(TMix):
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
                    cls_mask = mixup_mask["is_cls"]
                    cls_mask = torch.logical_and(cls_mask[:, :, None], cls_mask[mixup_indices][:, None, :])
                    sep_mask = mixup_mask["is_sep"]
                    sep_mask = torch.logical_and(sep_mask[:, :, None], sep_mask[mixup_indices][:, None, :])
                    normal_mask = mixup_mask["is_normal"]
                    normal_mask = torch.logical_and(normal_mask[:, :, None], normal_mask[mixup_indices][:, None, :])
                    mixup_mask = cls_mask | sep_mask | normal_mask # [B, L1, L2]
                    with torch.no_grad():
                        model.mixup_mask = mixup_mask.detach()
                    q = module_dict["query"](h).view(batch, seq_len, self.n_head, self.k_dim)
                    k = module_dict["key"](h).view(batch, seq_len, self.n_head, self.k_dim)
                    q = q.permute(0, 2, 1, 3)
                    k = k.permute(0, 2, 3, 1)
                    # Challenge. Query vs Key? Query vs Query? Key vs Key?
                    cross_a = torch.matmul(q, k[mixup_indices]) / math.sqrt(self.k_dim) # [B, H, L1, L2]
                    cross_a = masked_softmax(cross_a, attention_mask[mixup_indices][:, None, None, :], dim=3)
                    # Challenge. Multi-head similarity? mean? max? approx?
                    cross_sim = torch.max(cross_a, dim=1)[0] # [B, L1, L2]
                    # Inspect similarity
                    with torch.no_grad():
                        self.cross_sim = cross_sim.detach()
                    mixup_position = masked_argmax(cross_sim, mixup_mask, dim=2, keepdim=True) # [B, L1, 1]
                    with torch.no_grad():
                        self.mixup_position = mixup_position.detach()
                h2 = torch.gather(input=h[mixup_indices], dim=1, index=mixup_position.expand(-1, -1, h.shape[2]))
                h = alpha * h + (1 - alpha) * h2
            h = self._forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return self.classifier(torch.mean(h, dim=1))


def evaluate(model, loader, step):
    with torch.no_grad():
        model.eval()
        with tqdm(loader, desc="Evaluate", ncols=200, leave=True, position=0) as test_tbar:
            correct, count = 0, 0
            for batch in test_tbar:
                input_ids = batch["inputs"]["input_ids"].to(device) 
                attention_mask = batch["inputs"]["attention_mask"].to(device) 
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = outputs.argmax(dim=1)
                correct += (labels == pred).float().sum()
                count += labels.shape[0]
                test_tbar.set_postfix(test_acc="%.4f" % (correct/count).item())
            acc = correct / count
            writer.add_scalar("test_acc", acc, global_step=step)
        model.train()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reproducibility parameter
    parser.add_argument("--seed", type=int, default=0)
    # Data hyperparameter
    parser.add_argument("--dataset", type=str, choices=["ag_news", "amazon_review_full", "yelp_polarity"], default="amazon_review_full")
    parser.add_argument("--num_train", type=int, default=-1, help="Number of train dataset. Use first `num_train` row. -1 means whole dataset")
    # Model hyperparameter
    parser.add_argument("--restore", type=str)
    # Train hyperparameter
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    # Train hyperparameter - augmentation
    parser.add_argument("--augment", type=str, choices=["none", "tmix", "adamix", "pdistmix"], default="none")
    parser.add_argument("--mixup_layer", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if args.dataset == "ag_news":
        train_dataset = AGNewsDataset(os.path.join("dataset", "ag_news_csv", "train.csv"), tokenizer, args.num_train)
        test_dataset = AGNewsDataset(os.path.join("dataset", "ag_news_csv", "test.csv"), tokenizer, -1)
    elif args.dataset == "amazon_review_full":
        train_dataset = AmazonReviewFullDataset(os.path.join("dataset", "amazon_review_full_csv", "train.csv"), tokenizer)
        test_dataset = AmazonReviewFullDataset(os.path.join("dataset", "amazon_review_full_csv", "test.csv"), tokenizer)
    elif args.dataset == "yelp_polarity":
        train_dataset = YelpPolarityDataset(os.path.join("dataset", "yelp_review_polarity_csv", "train.csv"), tokenizer)
        test_dataset = YelpPolarityDataset(os.path.join("dataset", "yelp_review_polarity_csv", "test.csv"), tokenizer)

    if args.augment == "none":
        model = create_bert_sentence_classification_model(n_class=train_dataset.n_class)
    elif args.augment == "tmix":
        model = create_tmix_bert_sentence_classification_model(n_class=train_dataset.n_class)
    elif args.augment == "adamix":
        model = create_adamix_bert_sentence_classification_model(n_class=train_dataset.n_class)
    elif args.augment == "pdistmix":
        raise NotImplementedError("Not supported")
        model = PdistMixBert(n_class=train_dataset.n_class, mixup_layer=args.mixup_layer)

    model.load()     # Load BERT pretrained weight
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CollateFn(tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=CollateFn(tokenizer))

    if args.augment == "adamix":
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.augment == "none":
        optimizers = [optim.AdamW(model.embedding_model.parameters(), lr=args.lr),
                      optim.AdamW(model.classifier.parameters(), lr=1e-3)]
    elif args.augment == "tmix":
        optimizers = [optim.AdamW(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.AdamW(model.classifier.parameters(), lr=1e-3)]
    elif args.augment == "adamix":
        optimizers = [optim.AdamW(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.AdamW(model.mix_model.policy_region_generator.parameters(), lr=1e-3),
                      optim.AdamW(model.mix_model.intrusion_classifier.parameters(), lr=1e-3),
                      optim.AdamW(model.classifier.parameters(), lr=1e-3)]

    schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, max(-(x-20000)/(20000-1000), 0)))
                  for optimizer in optimizers]
    scaler = torch.cuda.amp.GradScaler(enabled=True)

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

    step, best_acc = 0, 0
    model.to(device)
    for epoch in range(args.epoch):
        with tqdm(train_loader, desc="Epoch %d" % epoch, ncols=200) as tbar:
            for batch in tbar:
                with torch.cuda.amp.autocast(enabled=True):
                    input_ids = batch["inputs"]["input_ids"].to(device) 
                    attention_mask = batch["inputs"]["attention_mask"].to(device) 
                    if "mixup_mask" in batch["inputs"]:
                        mixup_mask = batch["inputs"]["mixup_mask"]
                        mixup_mask["is_cls"] = mixup_mask["is_cls"].to(device)
                        mixup_mask["is_sep"] = mixup_mask["is_sep"].to(device)
                        mixup_mask["is_normal"] = mixup_mask["is_normal"].to(device)
                    else:
                        mixup_mask = None
                    labels = batch["labels"].to(device)

                    if args.augment == "none":
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs, labels)
                        tbar.set_postfix(loss="%.4f" % loss.item())
                    elif args.augment in ["tmix", "adamix", "pdistmix"]:
                        mixup_indices = torch.randperm(input_ids.shape[0], device=device)
                        if args.augment in ["tmix", "pdistmix"]:
                            lambda_ = np.random.beta(args.alpha, args.alpha)
                            if lambda_ < 0.5:
                                lambda_ = 1 - lambda_
                            if args.augment == "tmix":
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, lambda_=lambda_)
                            else:
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, mixup_mask=mixup_mask, alpha=lambda_)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(outputs, labels[mixup_indices])
                            loss = lambda_ * loss1 + (1 - lambda_) * loss2
                            tbar.set_postfix(loss1="%.4f" % loss1, loss2="%.4f" % loss2, loss="%.4f" % loss.item())
                        elif args.augment in ["adamix"]:
                            outs, mix_outs, intr, mix_intr, gamma = model(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices)
                            loss1 = criterion(outs, labels).mean()
                            loss2 = (gamma * criterion(mix_outs, labels) + (1 - gamma) * criterion(mix_outs, labels[mixup_indices])).mean()
                            loss3 = criterion2(torch.cat((intr, mix_intr), dim=0),
                                               torch.cat((torch.zeros_like(intr), torch.ones_like(mix_intr)), dim=0))
                            loss = loss1 + loss2 + loss3
                            tbar.set_postfix(loss1="%.4f" % loss1, loss2="%.4f" % loss2, loss3="%.4f" % loss3, loss="%.4f" % loss, gamma="%.4f" % gamma.mean())
                writer.add_scalar("train_loss", loss, global_step=step)
                for optimizer in optimizers:
                    optimizer.zero_grad()
                scaler.scale(loss).backward()
                # Track gradient norm
                #with torch.no_grad():
                #    for k, v in model.named_parameters():
                #        writer.add_scalar("grad/"+k, v.grad.data.norm(2), global_step=step)
                for optimizer in optimizers:
                    scaler.step(optimizer)
                    scaler.update()
                for scheduler in schedulers:
                    scheduler.step()
                step += 1
                if step % args.eval_every == 0:
                    acc = evaluate(model, test_loader, step)
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({"epoch": epoch,
                                    "model": model.state_dict(),
                                    "optimizer": [optimizer.state_dict() for optimizer in optimizers],
                                    "scheduler": [scheduler.state_dict() for scheduler in schedulers]},
                                   "checkpoint_best.pt")

    writer.add_hparams(hparam_dict=vars(args), metric_dict={"test_acc": best_acc})
    writer.close()