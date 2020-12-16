import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


def get_cached_filepath(filepath, is_valid=None):
    if is_valid == None:
        return os.path.join('cache', filepath)
    elif is_valid == True:
        return os.path.join('cache', filepath + '_valid')
    elif is_valid == False:
        return os.path.join('cache', filepath + '_train')
    else:
        raise AttributeError('Invalid argument')


def preprocess(load_f, filepath, tokenizer):
    cached_filepath = os.path.join('cache', 'cache_'+filepath)
    if not os.path.exists(cached_filepath):
        data = load_f(filepath)
        for row in tqdm(data, desc="Tokenize amazon text"):
            row["input"] = ' '.join(map(str, tokenizer(row["input"], max_length=512, truncation=True)["input_ids"]))
        os.makedirs(os.path.dirname(cached_filepath), exist_ok=True)
        save_csv(cached_filepath, data, ["input", "label"])
    return [{"input": list(map(int, row["input"].split(' '))), "label": int(row["label"])}
            for row in load_csv(cached_filepath)]


def load_ag_news(filepath):
    return [{"label": int(row["class"]) - 1, "input": row["title"]}
            for row in tqdm(load_csv(filepath, ["class", "title", "description"]), desc="Load ag news dataset")]


def load_amazon_review_full(filepath):
    return [{"label": int(row["class"]) - 1, "input": row["text"]}
            for row in tqdm(load_csv(filepath, ["class", "title", "text"]), desc="Load amazon dataset")]


def load_yelp_polarity(filepath):
    data = [{"label": int(row["class"]) - 1, "input": row["review"]}
            for row in tqdm(load_csv(filepath, ["class", "review"]), desc="Load yelp polarity")]
    return data


def create_dataset(dataset, filepath, tokenizer):
    if dataset == "ag_news":
        data_load_func = load_ag_news
        n_class = 4
    elif dataset == "amazon_review_full":
        data_load_func = load_amazon_review_full
        n_class = 5
    elif dataset == "yelp_polarity":
        data_load_func = load_yelp_polarity
        n_class = 2
    else:
        raise AttributeError("Invalid dataset")
    data = preprocess(data_load_func, filepath, tokenizer)
    return ListDataset(data, n_class)
    
    

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

