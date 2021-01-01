import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from data import create_train_and_valid_dataset
from data import create_test_dataset
from data import CollateFn
from model import create_model
from main import evaluate


def calculate_normal_loss(model, criterion, input_ids, attention_mask, labels, epoch, step):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs, labels)
    logging.info("[Epoch %d, Step %d] Loss: %.4f" % (epoch, step, loss))
    return loss


def calculate_tmix_loss(model, criterion, input_ids, attention_mask, labels, alpha, epoch, step):
    mixup_indices = torch.randperm(input_ids.shape[0], device=input_ids.device)
    lambda_ = np.random.beta(alpha, alpha)
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    mixup_indices=mixup_indices,
                    lambda_=lambda_)
    loss1 = criterion(outputs, labels)
    loss2 = criterion(outputs, labels[mixup_indices])
    loss = lambda_ * loss1 + (1 - lambda_) * loss2
    logging.info("[Epoch %d, Step %d] Lambda: %.4f, Loss1: %.4f, Loss2: %.4f, Loss: %.4f" % \
        (epoch, step, lambda_, loss1, loss2, loss))
    return loss


def run(args):
    logging.basicConfig(filename="./log", level=logging.INFO)
    args = argparse.Namespace(**args)

    # Dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_dataset, valid_dataset = create_train_and_valid_dataset(
        dataset=args.dataset, dirpath=args.data_dir, tokenizer=tokenizer,
        num_train_data=args.num_train_data)

    # Loader
    collate_fn = CollateFn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False,
                              collate_fn=collate_fn)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = create_model(augment=args.augment, mixup_layer=args.mixup_layer,
                         n_class=train_dataset.n_class, n_layer=6, drop_prob=args.drop_prob)
    model.load()     # Load BERT pretrained weight
    model.to(device)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.augment == "none":
        optimizers = [optim.Adam(model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]
    elif args.augment == "tmix":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]

    # Scheduler
    schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, 1))
                  for optimizer in optimizers]

    step, best_acc, patience = 0, 0, 0
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad()
    for epoch in range(1, args.epoch+1):
        for batch in train_loader:
            input_ids = batch["inputs"]["input_ids"].to(device) 
            attention_mask = batch["inputs"]["attention_mask"].to(device) 
            labels = batch["labels"].to(device)
            if args.augment == "none":
                loss = calculate_normal_loss(model, criterion, input_ids, attention_mask, labels,
                                             epoch, step)
            elif args.augment == "tmix":
                loss = calculate_tmix_loss(model, criterion, input_ids, attention_mask, labels,
                                           args.alpha, epoch, step)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            for scheduler in schedulers:
                scheduler.step()
            
            step += 1
            if step % args.eval_every == 0:
                acc = evaluate(model, valid_loader, device)
                tune.report(accuracy=acc)
                if best_acc < acc:
                    best_acc = acc
                    patience = 0
                    torch.save(model.state_dict(), "./model.pth")
                else:
                    patience += 1
                    if patience == 10:
                        break
        if patience == 10:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reproducibility parameter
    #parser.add_argument("--seed", type=int, default=0)
    # Data hyperparameter
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["ag_news", "yahoo_answer"], default="ag_news")
    parser.add_argument("--num_train_data", type=int, default=-1, help="Number of train dataset. Use first `num_train` row. -1 means whole dataset")
    # Train hyperparameter
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=24)
    # Train hyperparameter - augmentation
    parser.add_argument("--augment", type=str, choices=["none", "tmix", "adamix", "proposed"], default="none")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Define search space
    search_space = {k: tune.grid_search([v]) for k, v in vars(args).items()}
    search_space.update({
        "lr": tune.qloguniform(1e-6, 1e-4, 5e-7),
        "drop_prob": tune.quniform(0, 1, 0.1),
        "mixup_layer": tune.randint(0, 6),
        "alpha": tune.quniform(0, 1, 0.1),
    })

    # Execute run
    ray.init(num_gpus=2)
    result = tune.run(run,
                      num_samples=10,
                      scheduler=ASHAScheduler(metric="accuracy", mode="max"),
                      resources_per_trial={"gpu": 1},
                      name="%s_%s" % (args.dataset, args.augment),
                      local_dir="/data/sh0416/ray_results",
                      config=search_space)
    logdir = result.get_best_logdir("accuracy", "max")
    logging.info("Best trial logdir: {}".format(logdir))

    # Evaluate on test dataset
    with open(os.path.join(logdir, "params.json")) as f:
        args = argparse.Namespace(**json.load(f))
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    test_dataset = create_test_dataset(dataset=args.dataset,
                                       dirpath=args.data_dir,
                                       tokenizer=tokenizer)
    collate_fn = CollateFn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(augment=args.augment, mixup_layer=args.mixup_layer,
                         n_class=test_dataset.n_class, n_layer=6, drop_prob=args.drop_prob)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(logdir, "model.pth")))

    test_acc = evaluate(model, test_loader, device)
    logging.info("Test accuracy: %.4f" % test_acc)