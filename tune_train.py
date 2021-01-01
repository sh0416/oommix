import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from data import create_train_and_valid_dataset
from data import CollateFn
from model import create_model
from main import evaluate

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
    optimizers = [optim.Adam(model.embedding_model.parameters(), lr=args.lr),
                  optim.Adam(model.classifier.parameters(), lr=1e-3)]

    # Scheduler
    schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, 1))
                  for optimizer in optimizers]

    step, best_acc = 0, 0
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad()
    for epoch in range(1, args.epoch+1):
        for batch in train_loader:
            input_ids = batch["inputs"]["input_ids"].to(device) 
            attention_mask = batch["inputs"]["attention_mask"].to(device) 
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            logging.info("Loss: %.4f" % loss.item())
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
                    torch.save(model.state_dict(), "./model.pth")


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

    # Define search space
    search_space = {k: tune.grid_search([v]) for k, v in vars(args).items()}
    search_space.update({
        "lr": tune.loguniform(1e-6, 1e-4),
        "drop_prob": tune.uniform(0, 1),
        "mixup_layer": tune.randint(0, 12),
        "alpha": tune.uniform(0, 1),
    })

    # Execute run
    ray.init(num_gpus=2)
    analysis = tune.run(run,
                        num_samples=200,
                        scheduler=ASHAScheduler(metric="accuracy", mode="max", grace_period=5),
                        resources_per_trial={"gpu": 1},
                        name="ag_news_none",
                        local_dir="/data/sh0416/ray_results",
                        config=search_space)
