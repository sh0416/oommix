import os
import sys
import json
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from data import create_train_and_valid_dataset
from data import create_test_dataset
from data import CollateFn
from model import create_model
from main import train, evaluate


def run(args):
    logging.basicConfig(filename="./log", level=logging.INFO)
    args = argparse.Namespace(**args)
    train(args, report_func=tune.report)


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
    parser.add_argument("--batch_size", type=int, default=12)
    # Train hyperparameter - augmentation
    parser.add_argument("--augment", type=str, choices=["none", "tmix", "adamix", "proposed"], default="none")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Define search space
    search_space = {k: tune.grid_search([v]) for k, v in vars(args).items()}
    #search_space.update({
    #    "lr_lm": tune.grid_search([5e-6, 1e-5, 5e-5, 1e-4]),
    #    "lr_cls": tune.grid_search([1e-5, 1e-4, 1e-3]),
    #})
    if args.augment == "tmix":
        search_space.update({
            "mixup_layer": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            "alpha": tune.grid_search([0.1, 0.2, 0.4, 0.8]),
            "coeff_intr": tune.grid_search([0]),
        })
    elif args.augment == "adamix":
        search_space.update({
            "mixup_layer": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            "alpha": tune.grid_search([0]),
            "coeff_intr": tune.grid_search([0.1, 0.3, 0.5, 0.7, 0.9]),
        })

    # Execute run
    ray.init(num_gpus=3)
    result = tune.run(run,
                      verbose=2,
                      resources_per_trial={"gpu": 1},
                      name="%s_%04d_%s" % (args.dataset, args.num_train_data, args.augment),
                      local_dir="/data/sh0416/ray_results",
                      config=search_space)
    logdir = result.get_best_logdir("best_accuracy", "max")
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
                         n_class=test_dataset.n_class, n_layer=12, drop_prob=args.drop_prob)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(logdir, "model.pth")))

    test_acc = evaluate(model, test_loader, device)
    logging.info("Test accuracy: %.4f" % test_acc)
