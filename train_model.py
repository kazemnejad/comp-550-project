import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from classifiers import BertClassifier, CNNClassifier
from utils import train, evaluate, split_train_valid, data_to_df
import torch
from torchtext.datasets import AG_NEWS
import wandb
import os
import sys

HYPERPARAMS = {
    "dataset": "AG_NEWS",
    "num_classes": 4,
    "valid_prop": 0.15,
    "epochs": 5,
    "lr": 1e-4,
    "batch_size": 32,
    "model": "cnn",
    "seed": 0,
    "wandb": False,
    # cnn hyperparams
    "num_layers": 3,
    "hidden_dim": 256,
    "kernel_size": 10,
}

SWEEP_CONFIG = {
    "name": "cnn-sweep",
    "method": "grid",
    "parameters": {
        "num_layers": {"values": [2, 4, 6, 8]},
        "hidden_dim": {"values": [128, 180, 220, 256]},
        "kernel_size": {"values": [3, 7, 11]},
    },
}

MODELS = {
    "cnn": lambda hyperparams: CNNClassifier(hyperparams),
    "transformer": lambda hyperparams: BertClassifier(hyperparams),
}

torch.manual_seed(HYPERPARAMS["seed"])
np.random.seed(HYPERPARAMS["seed"])


def train_model():
    if HYPERPARAMS["wandb"]:
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="train-model",
        )
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="train-model",
        )

    train_iter, _ = AG_NEWS(root=".data", split=("train", "test"))

    train_iter, valid_iter = split_train_valid(train_iter, HYPERPARAMS["valid_prop"])

    train_df = data_to_df(train_iter)
    valid_df = data_to_df(valid_iter)

    train_df.to_pickle("./data/train_df.pkl")
    valid_df.to_pickle("./data/valid_df.pkl")

    wandb.log_artifact("./data/train_df.pkl", name="train_dataset", type="dataset")
    wandb.log_artifact("./data/valid_df.pkl", name="valid_dataset", type="dataset")

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    embedding_model = BertModel.from_pretrained("bert-base-cased")

    model = MODELS[HYPERPARAMS["model"]](HYPERPARAMS)
    wandb.watch(model)

    train(model, embedding_model, train_df, valid_df, tokenizer, HYPERPARAMS)


def evaluate_model():
    _, test_iter = AG_NEWS(root=".data", split=("train", "test"))

    test_df = data_to_df(test_iter)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    embedding_model = BertModel.from_pretrained("bert-base-cased")

    model = MODELS[HYPERPARAMS["model"]](HYPERPARAMS)

    evaluate(model, embedding_model, test_df, tokenizer)


def create_sweep():
    wandb.sweep(
        SWEEP_CONFIG, entity="comp-555-project", project="classification_models"
    )


def run_agent(sweep_id):
    wandb.agent(
        sweep_id,
        function=train_model,
        entity="comp-555-project",
        project="classification_models",
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    print(mode)
    if not mode or mode == "train":
        train_model()
    elif mode == "sweep":
        create_sweep()
    elif mode == "agent":
        sweep_id = sys.argv[2]
        run_agent(sweep_id)
