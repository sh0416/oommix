import os
import csv
import logging
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class ListDataset(Dataset):
    def __init__(self, data, n_class):
        self.data = data
        self.n_class = n_class

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
        writer.writerows(data)


def get_cached_filepath(filepath):
    return os.path.join('cache', filepath)


def preprocess(load_f, filepath, tokenizer):
    cached_filepath = os.path.join('cache', 'cache_'+filepath)
    if not os.path.exists(cached_filepath):
        data = load_f(filepath)
        for row in tqdm(data, desc="Tokenize amazon text"):
            row["input"] = ' '.join(map(str, tokenizer(row["input"], max_length=256, truncation=True)["input_ids"]))
        os.makedirs(os.path.dirname(cached_filepath), exist_ok=True)
        save_csv(cached_filepath, data, ["input", "label"])
    return [{"input": list(map(int, row["input"].split(' '))), "label": int(row["label"])}
            for row in load_csv(cached_filepath)]


def load_ag_news(filepath):
    return [{"label": int(row["class"]) - 1, "input": row["description"]}
            for row in tqdm(load_csv(filepath, ["class", "title", "description"]), desc="Load ag news dataset")]


def load_yahoo_answer(filepath):
    return [{"label": int(row["class"]) - 1, "input": row["title"] + ' ' + row["content"] + ' ' + row["answer"]}
            for row in tqdm(load_csv(filepath, ["class", "title", "content", "answer"]), desc="Load yahoo dataset")]


def load_amazon_review_full(filepath):
    return [{"label": int(row["class"]) - 1, "input": row["text"]}
            for row in tqdm(load_csv(filepath, ["class", "title", "text"]), desc="Load amazon dataset")]


def load_yelp_polarity(filepath):
    data = [{"label": int(row["class"]) - 1, "input": row["review"]}
            for row in tqdm(load_csv(filepath, ["class", "review"]), desc="Load yelp polarity")]
    return data


def create_metadata(dataset):
    if dataset == "ag_news":
        data_load_func = load_ag_news
        n_class = 4
        num_valid_data = 1900 * n_class
    elif dataset == "yahoo_answer":
        data_load_func = load_yahoo_answer
        n_class = 10
        num_valid_data = 5000 * n_class
    elif dataset == "amazon_review_full":
        data_load_func = load_amazon_review_full
        n_class = 5
    elif dataset == "yelp_polarity":
        data_load_func = load_yelp_polarity
        n_class = 2
    else:
        raise AttributeError("Invalid dataset")
    return data_load_func, n_class, num_valid_data


def augment_data(data, split_num, reflection):
    pair_frequency = Counter()
    for row in data:
        pair_frequency.update(zip(row['input'], row['input'][1:]))
    pair_priority = {pair: i for i, (pair, count) in enumerate(pair_frequency.most_common())}
    augmented_data = []
    for row in tqdm(data):
        pairs = list(zip(row['input'], row['input'][1:]))
        priorities = list(map(lambda x: pair_priority[x], pairs))
        rank = np.asarray(priorities).argsort()[::-1].argsort()
        sublist, sid = [], 0
        for eid, rank in enumerate(rank, start=1):
            if rank < split_num:
                sublist.append(row['input'][sid:eid])
                sid = eid
        for _ in range(reflection):
            np.random.shuffle(sublist)
            augmented_data.append(
                {'input': [t for l in sublist for t in l],
                 'label': row['label']}
            )
    return data + augmented_data


def create_train_and_valid_dataset(dataset, dirpath, tokenizer, num_train_data=-1):
    data_load_func, n_class, num_valid_data = create_metadata(dataset)
    train_data = preprocess(data_load_func, os.path.join(dirpath, "train.csv"), tokenizer)
    # Stratified split
    train_data, valid_data = train_test_split(train_data, test_size=num_valid_data, random_state=42,
                                              shuffle=True, stratify=[x["label"] for x in train_data])
    # For only valid data, sort by length to accelerate inference
    valid_data = sorted(valid_data, key=lambda x: len(x["input"]), reverse=True)
    if num_train_data != -1:
        _, train_data = train_test_split(train_data, test_size=num_train_data, random_state=42,
                                         shuffle=True, stratify=[x["label"] for x in train_data])

    # Calculate the observed token number
    train_token = set(token for row in train_data for token in row["input"])
    valid_token = set(token for row in valid_data for token in row["input"])
    oov_token = valid_token - train_token
    logging.info("Train observed token number: %d" % len(train_token))
    logging.info("Valid observed token number: %d" % len(valid_token))
    logging.info("Out of vocabulary token number: %d" % len(oov_token))
    logging.info("Ouf of vocabulary rate: %.4f" % (len(oov_token) / len(valid_token)))
    train_dataset = ListDataset(train_data, n_class)
    valid_dataset = ListDataset(valid_data, n_class)
    return train_dataset, valid_dataset
    

def create_test_dataset(dataset, dirpath, tokenizer):
    data_load_func, n_class, _ = create_metadata(dataset)
    test_data = preprocess(data_load_func, os.path.join(dirpath, "test.csv"), tokenizer)
    test_data = sorted(test_data, key=lambda x: len(x["input"]), reverse=True)
    test_dataset = ListDataset(test_data, n_class)
    return test_dataset


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

