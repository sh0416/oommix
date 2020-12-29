import os
import logging
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
#import wandb
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertConfig, BertTokenizerFast
import matplotlib
import matplotlib.pyplot as plt
from torchviz import make_dot

from data import create_test_dataset, CollateFn
from model import create_model

matplotlib.use('Agg')
#torch.set_printoptions(profile="full", linewidth=150)
#wandb.init(project='pdistmix')

config = BertConfig.from_pretrained("bert-base-uncased")
print(config)

def evaluate(model, loader):
    with torch.no_grad():
        model.eval()
        correct, count = 0, 0
        for batch in loader:
            input_ids = batch["inputs"]["input_ids"].to(device) 
            attention_mask = batch["inputs"]["attention_mask"].to(device) 
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs.argmax(dim=1)
            correct += (labels == pred).float().sum()
            count += labels.shape[0]
        acc = correct / count
        model.train()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reproducibility parameter
    parser.add_argument("--seed", type=int, default=0)
    # Data hyperparameter
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["ag_news", "yahoo_answer"], default="ag_news")
    parser.add_argument("--num_train_data", type=int, default=-1, help="Number of train dataset. Use first `num_train` row. -1 means whole dataset")
    # Model hyperparameter
    parser.add_argument("--restore", type=str)
    # Train hyperparameter
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    # Train hyperparameter - augmentation
    parser.add_argument("--augment", type=str, choices=["none", "tmix", "adamix", "proposed"], default="none")
    parser.add_argument("--mixup_layer", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("CUDA: %d" % torch.cuda.is_available())
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    test_dataset = create_test_dataset(dataset=args.dataset,
                                       dirpath=args.data_dir,
                                       tokenizer=tokenizer)
    logging.info("Test data: %d" % (len(test_dataset)))
    collate_fn = CollateFn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)

    model = create_model(augment=args.augment, mixup_layer=args.mixup_layer,
                         n_class=test_dataset.n_class, n_layer=6, drop_prob=args.drop_prob)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    if args.restore is not None:
        checkpoint = torch.load(args.restore)
        model.load_state_dict(checkpoint["model"])

    step, best_acc, patience = 0, 0, 0
    model.to(device)

    logging.info("Hyperparameter")
    for k, v in vars(args).items():
        logging.info("%s = %s" % (str(k), str(v)))
    
    logging.info("Test acc: %.4f" % (evaluate(model, test_loader)))
