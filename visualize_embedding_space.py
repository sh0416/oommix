import sys
import logging
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from data import create_test_dataset
from data import CollateFn
from model import create_model


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu().numpy()
    return hook


def probe(model, loader, device):
    np.set_printoptions(precision=4, linewidth=1e+8, suppress=True)
    df = pd.DataFrame()
    activations = defaultdict(list)

    with torch.no_grad():
        model.eval()
        correct, count = 0, 0
        for batch in loader:
            input_ids = batch["inputs"]["input_ids"].to(device) 
            attention_mask = batch["inputs"]["attention_mask"].to(device) 
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs.argmax(dim=1)
            # Create rows for this batch
            rows = {
                "context_idx": [i.item() for i in batch["idx"] for _ in range(batch["inputs"]["input_ids"].shape[1])],
                "token_idx": [x.item() for l in batch["inputs"]["input_ids"] for x in l],
            }
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            for name, t in activation.items():
                activations[name].append(np.reshape(t, (-1, t.shape[2])))

    df.to_csv("representations.csv", index=False)
    for name, l in activations.items():
        np.savetxt(name+'.csv', np.vstack(l), fmt="%.4f", delimiter=",", encoding='utf8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reproducibility parameter
    parser.add_argument("--seed", type=int, default=0)
    # Data hyperparameter
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["ag_news", "yahoo_answer"], default="ag_news")
    # Model hyperparameter
    parser.add_argument("--restore", type=str, required=True)
    # Train hyperparameter - augmentation
    parser.add_argument("--augment", type=str, choices=["none", "tmix", "adamix", "proposed"], default="none")
    parser.add_argument("--mixup_layer", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--coeff_intr", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    test_dataset = create_test_dataset(dataset=args.dataset,
                                       dirpath=args.data_dir,
                                       tokenizer=tokenizer)
    collate_fn = CollateFn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(augment=args.augment, mixup_layer=args.mixup_layer,
                         n_class=test_dataset.n_class, n_layer=12, drop_prob=0.0)
    model.to(device)
    model.load_state_dict(torch.load(args.restore))
    for name, module in model.mix_model.embedding_model.named_modules():
        if name in ["embedding_norm",
                    "encoder.0.norm", "encoder.0.ff_norm", "encoder.1.norm", "encoder.1.ff_norm",
                    "encoder.2.norm", "encoder.2.ff_norm", "encoder.3.norm", "encoder.3.ff_norm",
                    "encoder.4.norm", "encoder.4.ff_norm", "encoder.5.norm", "encoder.5.ff_norm",
                    "encoder.6.norm", "encoder.6.ff_norm", "encoder.7.norm", "encoder.7.ff_norm",
                    "encoder.8.norm", "encoder.8.ff_norm", "encoder.9.norm", "encoder.9.ff_norm",
                    "encoder.10.norm", "encoder.10.ff_norm", "encoder.11.norm", "encoder.11.ff_norm"]:
            module.register_forward_hook(get_activation(name))

    probe(model, test_loader, device)
