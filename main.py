import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertConfig, BertTokenizerFast
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from torchviz import make_dot

from data import create_dataset, CollateFn
from model import create_model

matplotlib.use('Agg')
#torch.set_printoptions(profile="full", linewidth=150)
#wandb.init(project='pdistmix')

config = BertConfig.from_pretrained("bert-base-uncased")
print(config)

def evaluate(model, loader, step):
    with torch.no_grad():
        model.eval()
        with tqdm(loader, desc="Evaluate", ncols=200) as test_tbar:
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
            writer.add_scalar("acc", acc, global_step=step)
        model.train()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reproducibility parameter
    parser.add_argument("--seed", type=int, default=0)
    # Data hyperparameter
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["ag_news"], default="ag_news")
    parser.add_argument("--num_train_data", type=int, default=-1, help="Number of train dataset. Use first `num_train` row. -1 means whole dataset")
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
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    print("CUDA: %d" % torch.cuda.is_available())
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_dataset = create_dataset(dataset=args.dataset,
                                   filepath=os.path.join(args.data_dir, "train.csv"),
                                   tokenizer=tokenizer)
    test_dataset = create_dataset(dataset=args.dataset,
                                  filepath=os.path.join(args.data_dir, "test.csv"),
                                  tokenizer=tokenizer)
    train_num, valid_num = int(0.9*len(train_dataset)), len(train_dataset)-int(0.9*len(train_dataset))
    train_dataset, valid_dataset = random_split(train_dataset,
                                                lengths=[train_num, valid_num],
                                                generator=torch.Generator().manual_seed(42))
    collate_fn = CollateFn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = create_model(augment=args.augment, n_class=test_dataset.n_class, n_layer=6)
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
    for optimizer in optimizers:
        optimizer.zero_grad()
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
                        scaler.scale(loss).backward()
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
                            scaler.scale(loss).backward()
                        elif args.augment in ["adamix"]:
                            eps = torch.rand(input_ids.shape[0], device=input_ids.device)
                            # Calculate gradient for normal classifier
                            outs, mix_outs, gamma = model(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, eps=eps)
                            loss1 = criterion(outs, labels).mean()
                            loss2 = (gamma * criterion(mix_outs, labels) + (1 - gamma) * criterion(mix_outs, labels[mixup_indices])).mean()
                            loss = loss1 + loss2
                            #dot = make_dot(loss, params=dict(model.named_parameters()))
                            #dot.format = 'png'
                            #dot.render("gradient_tree1")
                            scaler.scale(loss).backward()
                            # Calculate gradient for intrusion classifier
                            for param in model.mix_model.embedding_model.parameters():
                                param.requires_grad = False
                            """
                            for name, param in model.mix_model.embedding_model.named_parameters():
                                if 'encoder.4.ff_norm.bias' in name:
                                    print()
                                    print()
                                    print()
                                    print(name, param.grad)
                            """
                            outs, mix_outs = model.predict_intrusion(input_ids=input_ids, attention_mask=attention_mask, mixup_indices=mixup_indices, eps=eps)
                            intr_loss = criterion2(torch.cat((outs, mix_outs), dim=0),
                                                   torch.cat((torch.ones_like(outs), torch.zeros_like(mix_outs)), dim=0))
                            #dot = make_dot(intr_loss, params=dict(model.named_parameters()))
                            #dot.format = 'png'
                            #dot.render("gradient_tree2")
                            #for k, param in model.mix_model.policy_region_generator.named_parameters():
                            #    if k == "2.bias":
                            #        print('before', k, param.grad)
                            scaler.scale(10*intr_loss).backward()
                            #for k, param in model.mix_model.policy_region_generator.named_parameters():
                            #    if k == "2.bias":
                            #        print('after', k, param.grad)
                            """
                            for name, param in model.mix_model.embedding_model.named_parameters():
                                if 'encoder.4.ff_norm.bias' in name:
                                    print()
                                    print()
                                    print()
                                    print(name, param.grad)
                            """
                            for name, param in model.mix_model.embedding_model.named_parameters():
                                if 'word' not in name:
                                    param.requires_grad = True
                            #tbar.set_postfix(loss="%.4f" % loss, intr_loss="%.4f" % intr_loss, gamma="%.4f" % gamma.mean())
                            #wandb.log({"Gamma": gamma.mean()})
                writer.add_scalar("train_loss", loss, global_step=step)
                #wandb.log({"Train loss": loss})
                # Track gradient norm
                #with torch.no_grad():
                #    for k, v in model.named_parameters():
                #        writer.add_scalar("grad/"+k, v.grad.data.norm(2), global_step=step)
                for optimizer in optimizers:
                    scaler.step(optimizer)
                    scaler.update()
                for scheduler in schedulers:
                    scheduler.step()
                for optimizer in optimizers:
                    optimizer.zero_grad()
                step += 1
                if step % args.eval_every == 0:
                    train_acc = evaluate(model, train_loader, step)
                    valid_acc = evaluate(model, valid_loader, step)
                    print("train_accuracy: %.4f, valid_accuracy: %.4f" % (train_acc, valid_acc))
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        test_acc = evaluate(model, test_loader, step)
                        print("Best valid accuracy! test accuracy: %.4f" % (test_acc))
                        torch.save({"epoch": epoch,
                                    "model": model.state_dict(),
                                    "optimizer": [optimizer.state_dict() for optimizer in optimizers],
                                    "scheduler": [scheduler.state_dict() for scheduler in schedulers]},
                                   "checkpoint_best.pt")
                        #writer.add_scalars('acc', {"train": train_acc, "valid": valid_acc, "test": test_acc}, step)
                        #wandb.log({"Train acc": train_acc, "Valid acc": valid_acc, "Test acc": test_acc})
                    else:
                        pass
                        #writer.add_scalars('acc', {"train": train_acc, "valid": valid_acc}, step)
                        #wandb.log({"Train acc": train_acc, "Valid acc": valid_acc})

    writer.add_hparams(hparam_dict=vars(args), metric_dict={"test_acc": test_acc})
    writer.close()
