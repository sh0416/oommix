import os
import sys
import json
import uuid
import logging
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from transformers import BertModel, BertConfig, BertTokenizerFast
import matplotlib
import matplotlib.pyplot as plt
from torchviz import make_dot
from tqdm import tqdm

from data import create_train_and_valid_dataset, CollateFn
from data import create_test_dataset
from model import create_model
from utils import Collector

matplotlib.use('Agg')
#torch.set_printoptions(profile="full", linewidth=150)
#wandb.init(project='pdistmix')

config = BertConfig.from_pretrained("bert-base-uncased")


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


def calculate_adamix_loss(model, criterion, input_ids, attention_mask,
                          labels, mixup_indices, eps, epoch, step):
    outs, mix_outs, gamma, intr_loss = model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             mixup_indices=mixup_indices,
                                             eps=eps)
    loss1 = criterion(outs, labels).mean()
    loss21 = criterion(mix_outs, labels)
    loss22 = criterion(mix_outs, labels[mixup_indices])
    loss2 = (gamma * loss21 + (1 - gamma) * loss22).mean()
    loss = (loss1 + loss2) / 2
    logging.info("[Epoch %d, Step %d] Loss: %.4f" % (epoch, step, loss))
    logging.info("[Epoch %d, Step %d] Intrusion loss: %.4f" % (epoch, step, intr_loss))
    return loss, intr_loss


def plot_representation(model, loader, writer, device, step):
    collector = Collector()
    collector.collect_representation(model.get_embedding_model())
    collector.collect_attention(model.get_embedding_model())
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
            for i in range(12):
                k = "encoder.%d.mhsa_attn" % i
                t = collector.activations[k]
                topk, _ = torch.topk(t, 5, dim=3)
                topk_mask = t >= topk[:, :, :, [-1]]
                mask = attention_mask.view(-1, 1, 1, attention_mask.shape[1]).expand_as(t).cpu()
                topk_mask = topk_mask.view(-1, 1, topk_mask.shape[2], topk_mask.shape[3])
                t = t.view(-1, 1, t.shape[2], t.shape[3])
                mask = mask.reshape(-1, 1, t.shape[2], t.shape[3])
                t = torch.cat((topk_mask.float(), 1-mask.float(), torch.zeros_like(t)), dim=1)
                writer.add_image("vis/%s" % k, make_grid(t, nrow=12, pad_value=0.5), step)
            break
        model.train()
    collector.remove_all_hook()


def evaluate(model, loader, device):
    with torch.no_grad():
        model.eval()
        correct, count = 0, 0
        for batch in tqdm(loader):
            input_ids = batch["inputs"]["input_ids"].to(device) 
            attention_mask = batch["inputs"]["attention_mask"].to(device) 
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs.argmax(dim=1)
            correct += (labels == pred).float().sum()
            count += labels.shape[0]    
        acc = correct / count
        model.train()
    return acc.item()


def train(args, report_func=None):
    # Dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_dataset, valid_dataset = create_train_and_valid_dataset(
        dataset=args.dataset, dirpath=args.data_dir, tokenizer=tokenizer,
        num_train_data=args.num_train_data)

    # Loader
    collate_fn = CollateFn(tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False,
                              collate_fn=collate_fn)
    plot_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False,
                             collate_fn=collate_fn)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = create_model(augment=args.augment, mixup_layer=args.mixup_layer,
                         intrusion_layer=args.intrusion_layer,
                         n_class=train_dataset.n_class, n_layer=12, drop_prob=args.drop_prob)
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
    elif args.augment == "shufflemix":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]
    elif args.augment == "adamix":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.mix_model.policy_region_generator.parameters(), lr=1e-3),
                      optim.Adam(model.mix_model.intrusion_classifier.parameters(), lr=1e-4),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]

    # Scheduler
    schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, 1))
                  for optimizer in optimizers]

    # Writer
    writer = SummaryWriter(os.path.join("run", args.exp_id))

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
                loss.backward()
            elif args.augment == "tmix":
                loss = calculate_tmix_loss(model, criterion, input_ids, attention_mask, labels,
                                           args.alpha, epoch, step)
                loss.backward()
            elif args.augment == "shufflemix":
                loss = calculate_tmix_loss(model, criterion, input_ids, attention_mask, labels,
                                           args.alpha, epoch, step)
                loss.backward()
            elif args.augment == "adamix":
                mixup_indices = torch.randperm(input_ids.shape[0], device=device)
                eps = torch.rand(input_ids.shape[0], device=device)
                loss, intr_loss = calculate_adamix_loss(model, criterion, input_ids,
                        attention_mask, labels, mixup_indices, eps, epoch, step)
                #g = make_dot(loss, params=dict(model.named_parameters()))
                #g.render('graph_loss', 'graph', format='png')
                #g = make_dot(intr_loss, params=dict(model.named_parameters()))
                #g.render('graph_intr_loss', 'graph', format='png')
                # Order is important! Update intrusion parameter and 
                (args.coeff_intr*intr_loss).backward(retain_graph=True)
                #g = make_dot(loss, params=dict(model.named_parameters()))
                #g.render('graph_loss_after_backward', 'graph', format='png')
                optimizers[0].zero_grad()
                # Order is important! Update model
                ((1-args.coeff_intr)*loss).backward()
                for name, p in model.named_parameters():
                    if p.requires_grad == True:
                        value = 0 if p.grad is None else p.grad.data.norm(2)
                        writer.add_scalar("grad/%s" % name, value, step)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            for scheduler in schedulers:
                scheduler.step()
            
            step += 1
            if step % 50 == 0:
                #plot_representation(model, plot_loader, writer, device, step)
                pass
            if step % args.eval_every == 0:
                acc = evaluate(model, valid_loader, device)
                if best_acc < acc:
                    best_acc = acc
                    patience = 0
                    os.makedirs("ckpt", exist_ok=True)
                    torch.save(model.state_dict(), os.path.join("ckpt", "./model_%s.pth" % args.exp_id))
                else:
                    patience += 1
                    if patience == args.patience:
                        break
                logging.info("Accuracy: %.4f, Best accuracy: %.4f" % (acc, best_acc))
                if report_func is not None:
                    report_func(accuracy=acc, best_accuracy=best_acc)
        if patience == args.patience:
            break
    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reproducibility parameter
    parser.add_argument("--seed", type=int, default=0)
    # Data hyperparameter
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["ag_news", "yahoo_answer"], default="ag_news")
    parser.add_argument("--num_train_data", type=int, default=-1, help="Number of train dataset. Use first `num_train` row. -1 means whole dataset")
    parser.add_argument("--max_length", type=int, default=256)
    # Model hyperparameter
    parser.add_argument("--restore", type=str)
    # Train hyperparameter
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    # Train hyperparameter - augmentation
    parser.add_argument("--augment", type=str, choices=["none", "tmix", "shufflemix", "adamix"], default="none")
    parser.add_argument("--mixup_layer", type=int, default=3)
    parser.add_argument("--intrusion_layer", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--coeff_intr", type=float, default=0.5)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    args.exp_id = str(uuid.uuid4())[:8]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s - %(pathname)s - %(asctime)s - %(message)s")
    train(args)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    test_dataset = create_test_dataset(dataset=args.dataset,
                                       dirpath=args.data_dir,
                                       tokenizer=tokenizer)
    collate_fn = CollateFn(tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(augment=args.augment, mixup_layer=args.mixup_layer,
                         intrusion_layer=args.intrusion_layer,
                         n_class=test_dataset.n_class, n_layer=12, drop_prob=args.drop_prob)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join("ckpt", "model_%s.pth" % args.exp_id)))

    test_acc = evaluate(model, test_loader, device)
    logging.info("Test accuracy: %.4f" % test_acc)

    os.makedirs("param", exist_ok=True)
    with open(os.path.join("param", args.exp_id+".json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
