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

from data import create_train_and_valid_dataset, CollateFn
from data import create_test_dataset
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

    os.makedirs("log", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)
    experiment_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    ckpt_fpath = os.path.join('ckpt', '%s_best.pt' % experiment_id)
    logging.basicConfig(handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join('log', experiment_id))],
                        level=logging.INFO)

    logging.info("CUDA: %d" % torch.cuda.is_available())
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_dataset, valid_dataset = create_train_and_valid_dataset(
        dataset=args.dataset, dirpath=args.data_dir, tokenizer=tokenizer, num_train_data=args.num_train_data,
        augment=args.augment)
    test_dataset = create_test_dataset(dataset=args.dataset,
                                       dirpath=args.data_dir,
                                       tokenizer=tokenizer)
    logging.info("Train data: %d" % (len(train_dataset)))
    logging.info("Valid data: %d" % (len(valid_dataset)))
    logging.info("Test data: %d" % (len(test_dataset)))
    collate_fn = CollateFn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)

    model = create_model(augment=args.augment, mixup_layer=args.mixup_layer,
                         n_class=train_dataset.n_class, n_layer=6, drop_prob=args.drop_prob)
    model.load()     # Load BERT pretrained weight
    #wandb.watch(model)

    if args.augment == "adamix":
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.augment == "none":
        optimizers = [optim.Adam(model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]
    elif args.augment == "proposed":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]
    elif args.augment == "tmix":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]
    elif args.augment == "adamix":
        optimizers = [optim.Adam(model.mix_model.embedding_model.parameters(), lr=args.lr),
                      optim.Adam(model.mix_model.policy_region_generator.parameters(), lr=1e-3),
                      optim.Adam(model.intrusion_classifier.parameters(), lr=1e-3),
                      optim.Adam(model.classifier.parameters(), lr=1e-3)]

    #schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, max(-(x-20000)/(20000-1000), 0)))
    #              for optimizer in optimizers]
    schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x/1000, 1))
                  for optimizer in optimizers]
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    #writer = SummaryWriter()

    if args.restore is not None:
        checkpoint = torch.load(args.restore)
        model.load_state_dict(checkpoint["model"])
        for optimizer, restored_state_dict in zip(optimizers, checkpoints["optimizers"]):
            optimizer.load_state_dict(restored_state_dict)
        for scheduler, restored_state_dict in zip(schedulers, checkpoints["schedulers"]):
            scheduler.load_state_dict(restored_state_dict)

    step, best_acc, patience = 0, 0, 0
    model.to(device)

    logging.info("Hyperparameter")
    for k, v in vars(args).items():
        logging.info("%s = %s" % (str(k), str(v)))
    
    for optimizer in optimizers:
        optimizer.zero_grad()
    for epoch in range(1, args.epoch+1):
        for batch in train_loader:
            with torch.cuda.amp.autocast(enabled=False):
                input_ids = batch["inputs"]["input_ids"].to(device) 
                attention_mask = batch["inputs"]["attention_mask"].to(device) 
                labels = batch["labels"].to(device)

                if args.augment == "none":
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    logging.info("Loss: %.4f" % loss.item())
                elif args.augment == "proposed":
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    print(model.mix_model.embedding_model.word_embeddings.weight.grad)
                    """
                    mixup_indices = torch.randperm(input_ids.shape[0], device=device)
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, lambda_=1)
                        w = model.classifier.weight.data  # [C, K]
                        f = model.h  # [B, L, K]
                        class_score = torch.matmul(f, w.transpose(0, 1)) # [B, L, C]
                        class_score = torch.gather(class_score, dim=2, index=labels[:, None, None].expand(-1, class_score.shape[1], -1)).squeeze(2) # [B, L]
                        example = list(zip(tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist()),
                                            class_score[0, :, labels[0]].cpu().tolist()))
                        example = [(idx, word, score) for idx, (word, score) in enumerate(example)]
                        example = sorted(example, key=lambda x: x[2], reverse=True)
                        logging.info("Sentence:")
                        for idx, word, score in example:
                            logging.info("Position: %d, Word: %s, Score: %.4f" % (idx, word, score))
                        from model import masked_softmax
                        class_score = masked_softmax(class_score, mask=attention_mask, dim=1)
                        idx = torch.argsort(class_score, dim=1, descending=True)
                    lambda_ = np.random.beta(args.alpha, args.alpha)
                    outputs = model(input_ids=input_ids, idx=idx,
                                    attention_mask=attention_mask, mixup_indices=mixup_indices, lambda_=lambda_)
                    loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    """
                    logging.info("Loss: %.4f" % loss.item())
                elif args.augment in ["tmix", "adamix"]:
                    mixup_indices = torch.randperm(input_ids.shape[0], device=device)
                    if args.augment in ["tmix"]:
                        lambda_ = np.random.beta(args.alpha, args.alpha)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, lambda_=lambda_)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(outputs, labels[mixup_indices])
                        loss = lambda_ * loss1 + (1 - lambda_) * loss2
                        scaler.scale(loss).backward()
                        logging.info("[Epoch %d, Step %d] Lambda: %.4f, Loss1: %.4f, Loss2: %.4f, Loss: %.4f" % (epoch, step, lambda_, loss1, loss2, loss))
                    elif args.augment in ["adamix"]:
                        eps = torch.rand(input_ids.shape[0], device=input_ids.device)
                        # Calculate gradient for normal classifier
                        outs, mix_outs, gamma = model(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, eps=eps)
                        loss1 = criterion(outs, labels).mean()
                        loss2 = (gamma * criterion(mix_outs, labels) + (1 - gamma) * criterion(mix_outs, labels[mixup_indices])).mean()
                        loss = (loss1 + loss2) / 2
                        scaler.scale(loss).backward()
                        # Calculate gradient for intrusion classifier
                        for param in model.mix_model.embedding_model.parameters():
                            param.requires_grad = False
                        outs, mix_outs = model.predict_intrusion(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, eps=eps)
                        intr_loss = criterion2(torch.cat((outs, mix_outs), dim=0),
                                                torch.cat((torch.ones_like(outs), torch.zeros_like(mix_outs)), dim=0))
                        scaler.scale(10*intr_loss).backward()
                        for name, param in model.mix_model.embedding_model.named_parameters():
                            if 'word' not in name:
                                param.requires_grad = True
            for optimizer in optimizers:
                scaler.step(optimizer)
                scaler.update()
            for scheduler in schedulers:
                scheduler.step()
            for optimizer in optimizers:
                optimizer.zero_grad()
            step += 1
            if step % args.eval_every == 0:
                logging.info("Evaluate model")
                valid_acc = evaluate(model, valid_loader)
                if valid_acc > best_acc:
                    logging.info("Best valid accuracy: %.4f" % (valid_acc))
                    best_acc = valid_acc
                    torch.save({"epoch": epoch,
                                "model": model.state_dict(),
                                "optimizer": [optimizer.state_dict() for optimizer in optimizers],
                                "scheduler": [scheduler.state_dict() for scheduler in schedulers]},
                                ckpt_fpath)
                    patience = 0
                else:
                    logging.info("Valid accuracy: %.4f" % (valid_acc))
                    patience += 1
                    if patience == 5:
                        break
        if patience == 5:
            break

    # Restore best valid parameter
    checkpoint = torch.load(ckpt_fpath)
    model.load_state_dict(checkpoint["model"])
    logging.info("Test acc: %.4f" % (evaluate(model, test_loader)))
    #writer.add_hparams(hparam_dict=vars(args), metric_dict={"test_acc": test_acc})
    #writer.close()
