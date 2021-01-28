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
#from torchviz import make_dot
from tqdm import tqdm
from data import create_train_and_valid_dataset, CollateFn
from data import create_test_dataset
from model import create_model
from utils import Collector, gram_schmidt

matplotlib.use('Agg')
#torch.set_printoptions(profile="full", linewidth=150)

config = BertConfig.from_pretrained("bert-base-uncased")


def calculate_normal_loss(model, criterion, input_ids, attention_mask, labels, epoch, step):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs, labels)
    if step % 10 == 0:
        logging.info("[Epoch %d, Step %d] Loss: %.4f" % (epoch, step, loss))
    return loss


def calculate_tmix_loss(model, criterion, input_ids, attention_mask, labels, alpha, epoch, step):
    mixup_indices = torch.randperm(input_ids.shape[0], device=input_ids.device)
    lambda_ = np.random.beta(alpha, alpha)
    lambda_ = torch.tensor(np.where(lambda_>=0.5, lambda_, 1-lambda_), dtype=torch.float, device=input_ids.device)
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    mixup_indices=mixup_indices,
                    lambda_=lambda_)
    n_class = outputs.shape[1]
    labels = (labels[:, None] == torch.arange(n_class, device=labels.device).view(1, n_class)).float()
    labels = lambda_ * labels + (1 - lambda_) * labels[mixup_indices]
    loss = criterion(F.log_softmax(outputs, dim=1), labels)
    if step % 10 == 0:
        logging.info("[Epoch %d, Step %d] Lambda: %.4f, Loss: %.4f" % (epoch, step, lambda_, loss))
    return loss


def calculate_nonlinearmix_loss(model, criterion, input_ids, attention_mask, labels, alpha, epoch, step):
    mixup_indices = torch.randperm(input_ids.shape[0], device=input_ids.device)
    lambda_ = np.random.beta(alpha, alpha,
            size=(input_ids.shape[0], input_ids.shape[1], model.get_embedding_model().embed_dim))
    lambda_ = torch.from_numpy(lambda_).float().to(input_ids.device)
    out, phi = model(input_ids, attention_mask, mixup_indices, lambda_) # [B, D_L]
    l1 = model.get_label_embedding(labels)
    l2 = model.get_label_embedding(labels[mixup_indices])
    mix_l = l1 * phi + (1 - phi) * l2   # [B, D_L]
    loss = -F.cosine_similarity(out, mix_l).mean()
    if step % 10 == 0:
        logging.info("[Epoch %d, Step %d] Loss: %.4f" % (epoch, step, loss))
    return loss


def calculate_mixuptransformer_loss(model, criterion, input_ids, attention_mask, labels, epoch, step):
    mixup_indices = torch.randperm(input_ids.shape[0], device=input_ids.device)
    lambda_ = 0.5
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    mixup_indices=mixup_indices,
                    lambda_=lambda_)
    n_class = outputs.shape[1]
    labels = (labels[:, None] == torch.arange(n_class, device=labels.device).view(1, n_class)).float()
    labels = lambda_ * labels + (1 - lambda_) * labels[mixup_indices]
    loss = criterion(F.log_softmax(outputs, dim=1), labels)
    if step % 10 == 0:
        logging.info("[Epoch %d, Step %d] Lambda: %.4f, Loss: %.4f" % (epoch, step, lambda_, loss))
    return loss


def calculate_oommix_loss(model, criterion, input_ids, attention_mask,
                          labels, mixup_indices, eps, epoch, step, writer):
    outs, mix_outs, gamma, mani_loss = model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             mixup_indices=mixup_indices,
                                             eps=eps)
    loss1 = criterion(outs, labels).mean()
    loss21 = criterion(mix_outs, labels)
    loss22 = criterion(mix_outs, labels[mixup_indices])
    loss2 = (gamma * loss21 + (1 - gamma) * loss22).mean()
    loss = (loss1 + loss2) / 2
    if step % 10 == 0:
        logging.info("[Epoch %d, Step %d] Loss: %.4f" % (epoch, step, loss))
        logging.info("[Epoch %d, Step %d] Manifold classification loss: %.4f" % (epoch, step, mani_loss))
    for g in gamma.tolist():
        writer.write("%d,%.4f\n" % (step, g))
    return loss, mani_loss


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
            pred = model.predict(input_ids=input_ids, attention_mask=attention_mask)
            #pred = outputs.argmax(dim=1)
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
        num_train_data=args.num_train_data, augmentation=args.data_augment)

    # Loader
    collate_fn = CollateFn(tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False,
                              collate_fn=collate_fn)
    plot_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False,
                             collate_fn=collate_fn)
    
    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = create_model(augment=args.mix_strategy, mixup_layer=args.m_layer,
                         d_layer=args.d_layer,
                         n_class=train_dataset.n_class, n_layer=12, drop_prob=args.drop_prob)
    model.load()     # Load BERT pretrained weight
    model.to(device)

    # Criterion
    if args.mix_strategy == "none":
        criterion = nn.CrossEntropyLoss()
    elif args.mix_strategy == "tmix":
        criterion = nn.KLDivLoss(reduction="batchmean")
    elif args.mix_strategy == "nonlinearmix":
        criterion = None
    elif args.mix_strategy == "mixuptransformer":
        criterion = nn.KLDivLoss(reduction="batchmean")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.mix_strategy == "none":
        optimizers = [optim.Adam(model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]
    elif args.mix_strategy == "tmix":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]
    elif args.mix_strategy == "nonlinearmix":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.mix_model.policy_mapping_f.parameters(), lr=2e-5),
                      optim.Adam(model.classifier.parameters(), lr=1e-3),
                      optim.Adam([model.label_matrix], lr=1e-3)]
    elif args.mix_strategy == "mixuptransformer":
        optimizers = [optim.Adam(model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]
    elif args.mix_strategy == "oommix":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.mix_model.embedding_generator.parameters(), lr=2e-5),
                      optim.Adam(model.mix_model.manifold_discriminator.parameters(), lr=2e-5),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]

    # Scheduler
    schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, 1))
                  for optimizer in optimizers]

    # Writer
    writer = SummaryWriter(os.path.join("out", "run", args.exp_id))
    logging.info("Tensorboard dir: %s" % os.path.join("out", "run", args.exp_id))

    # Gamma
    os.makedirs(os.path.join("out", "gamma"), exist_ok=True)
    writer2 = open(os.path.join("out", "gamma", "%s_gamma.csv" % args.exp_id), "w")

    step, best_acc, patience = 0, 0, 0
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad()
    for epoch in range(1, args.epoch+1):
        for batch in train_loader:
            input_ids = batch["inputs"]["input_ids"].to(device) 
            attention_mask = batch["inputs"]["attention_mask"].to(device) 
            labels = batch["labels"].to(device)
            if args.mix_strategy == "none":
                loss = calculate_normal_loss(model, criterion, input_ids, attention_mask, labels,
                                             epoch, step)
                loss.backward()
            elif args.mix_strategy == "tmix":
                loss = calculate_tmix_loss(model, criterion, input_ids, attention_mask, labels,
                                           args.alpha, epoch, step)
                loss.backward()
            elif args.mix_strategy == "nonlinearmix":
                loss = calculate_nonlinearmix_loss(model, criterion, input_ids, attention_mask, labels,
                                                   args.alpha, epoch, step)
                loss.backward()
            elif args.mix_strategy == "mixuptransformer":
                loss = calculate_mixuptransformer_loss(model, criterion, input_ids, attention_mask, labels,
                                                       epoch, step)
                loss.backward()
            elif args.mix_strategy == "oommix":
                mixup_indices = torch.randperm(input_ids.shape[0], device=device)
                eps = torch.rand(input_ids.shape[0], device=device)
                loss, intr_loss = calculate_oommix_loss(model, criterion, input_ids,
                        attention_mask, labels, mixup_indices, eps, epoch, step, writer2)
                # Order is important! Update discriminator parameter and 
                (args.coeff_intr*intr_loss).backward(retain_graph=True)
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
            if args.mix_strategy == "nonlinearmix":
                # Apply gram schmidt
                with torch.no_grad():
                    gs = torch.from_numpy(gram_schmidt(model.label_matrix.t().cpu().numpy())).t().to(device)
                    model.label_matrix.copy_(gs)
            
            step += 1
            if step % 50 == 0:
                #plot_representation(model, plot_loader, writer, device, step)
                pass
            if step % args.eval_every == 0:
                acc = evaluate(model, valid_loader, device)
                if best_acc < acc:
                    best_acc = acc
                    patience = 0
                    os.makedirs(os.path.join("out", "ckpt"), exist_ok=True)
                    torch.save(model.state_dict(), os.path.join("out", "ckpt", "./model_%s.pth" % args.exp_id))
                else:
                    patience += 1
                logging.info("Accuracy: %.4f, Best accuracy: %.4f" % (acc, best_acc))
                if patience == args.patience:
                    break
                if report_func is not None:
                    report_func(accuracy=acc, best_accuracy=best_acc)
        if patience == args.patience:
            break
    writer.close()
    writer2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reproducibility parameter
    parser.add_argument("--seed", type=int, default=0)
    # Data hyperparameter
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="ag_news",
                        choices=["ag_news", "yahoo_answer", "amazon_review_polarity", "dbpedia"])
    parser.add_argument("--num_train_data", type=int, default=-1,
                        help="Number of train dataset. Use first `num_train` row. -1 means whole dataset")
    parser.add_argument("--data_augment", type=str, default="none", 
                        choices=["none", "eda", "backtranslate", "ssmba"])
    parser.add_argument("--max_length", type=int, default=256)
    # Model hyperparameter
    parser.add_argument("--restore", type=str)
    # Train hyperparameter
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    # Train hyperparameter - augmentation
    parser.add_argument("--mix_strategy", type=str,
                        choices=["none", "tmix", "nonlinearmix", "mixuptransformer", "oommix"], default="none")
    parser.add_argument("--m_layer", type=int, default=3)
    parser.add_argument("--d_layer", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--coeff_intr", type=float, default=0.5)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.exp_id = str(uuid.uuid4())[:12]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(os.path.join("out", "log"), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        #filename=os.path.join("out", "log", args.exp_id),
                        format="%(levelname)s - %(pathname)s - %(asctime)s - %(message)s")
    #logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s - %(pathname)s - %(asctime)s - %(message)s")
    train(args)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    test_dataset = create_test_dataset(dataset=args.dataset,
                                       dirpath=args.data_dir,
                                       tokenizer=tokenizer)
    collate_fn = CollateFn(tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(augment=args.mix_strategy, mixup_layer=args.m_layer,
                         d_layer=args.d_layer,
                         n_class=test_dataset.n_class, n_layer=12, drop_prob=args.drop_prob)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join("out", "ckpt", "model_%s.pth" % args.exp_id)))

    test_acc = evaluate(model, test_loader, device)
    for k, v in vars(args).items():
        logging.info("Parameter %s = %s" % (k, str(v)))
    logging.info("Test accuracy: %.4f" % test_acc)

    os.makedirs(os.path.join("out", "param"), exist_ok=True)
    with open(os.path.join("out", "param", args.exp_id+".json"), 'w') as f:
        args = vars(args)
        args["test_acc"] = test_acc
        json.dump(args, f, indent=2)
