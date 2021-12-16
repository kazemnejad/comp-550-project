import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from classifiers import BertClassifier, CNNClassifier
from utils import train, evaluate, split_train_valid, data_to_df
import torch
from torchtext.datasets import AG_NEWS
import wandb
import os

HYPERPARAMS = {
    "dataset": "AG_NEWS",
    "num_classes": 4,
    "valid_prop": 0.15,
    "epochs": 5,
    "lr": 1e-4,
    "batch_size": 2,
    "model": "cnn",
    "seed": 0,
    "wandb": True,
    # cnn hyperparams
    "num_layers": 3,
    "hidden_dim": 256,
    "kernel_size": 10,
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


if __name__ == "__main__":
    train_model()
