import numpy as np
from classifiers import (
    BertClassifier,
    CNNClassifier,
    LSTMClassifier,
    TransformerClassifier,
)
from utils import train, evaluate, split_train_valid, get_dataset
import torch
from torchtext.datasets import AG_NEWS, YelpReviewFull
import wandb
import os
import sys

HYPERPARAMS = {
    "dataset": "AG_NEWS",
    "tokenizer": "basic_english",
    "max_len": 512,
    "num_classes": 5,
    "valid_prop": 0.15,
    "epochs": 30,
    "lr": 5e-3,
    "weight_decay": 1e-10,
    "batch_size": 16,
    "vocab_size": 100000,
    "embedding_size": 32,
    "model": "lstm",
    "seed": 0,
    "wandb": False,
    # cnn hyperparams
    "num_layers": 3,
    "hidden_dim": 256,
    "kernel_size": 3,
    # transformer hyperparams
    "num_heads": 4,
    "num_layers": 3,
    # lstm hyperparams
    "num_layers": 3,
    "hidden_dim": 256,
}

SWEEP_CONFIG = {
    "name": "cnn-sweep-hyperparams-yelp",
    "method": "grid",
    "parameters": {
        "num_layers": {"values": [2, 4, 6, 8]},
        "hidden_dim": {"values": [128, 200, 256]},
        "kernel_size": {"values": [3, 7, 11]},
        "embedding_size": {"values": [32, 64, 128]},
        # "lr": {"values": [1e-2, 5e-2, 1e-3, 5e-3]}
    },
}


MODELS = {
    "cnn": lambda hyperparams: CNNClassifier(hyperparams),
    "transformer": lambda hyperparams: TransformerClassifier(hyperparams),
    "lstm": lambda hyperparams: LSTMClassifier(hyperparams),
}

DATASETS = {
    "AG_NEWS": lambda: AG_NEWS(root=".data", split=("train", "test")),
    "Yelp": lambda: YelpReviewFull(root=".data", split=("train", "test")),
}

torch.manual_seed(HYPERPARAMS["seed"])
np.random.seed(HYPERPARAMS["seed"])


def generate_dataset():
    if HYPERPARAMS["wandb"]:
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="generate-dataset",
        )
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="generate-dataset",
        )

    train_iter, test_iter = DATASETS[HYPERPARAMS["dataset"]]()
    train_iter, valid_iter = split_train_valid(train_iter, HYPERPARAMS["valid_prop"])

    train_dataset, valid_dataset, test_dataset = get_dataset(
        train_iter,
        valid_iter,
        test_iter,
        HYPERPARAMS["max_len"],
        HYPERPARAMS["tokenizer"],
    )

    torch.save(train_dataset, f"./data/train_dataset_{HYPERPARAMS['dataset']}.pt")
    torch.save(valid_dataset, f"./data/valid_dataset_{HYPERPARAMS['dataset']}.pt")
    torch.save(test_dataset, f"./data/test_dataset_{HYPERPARAMS['dataset']}.pt")

    wandb.log_artifact(
        f"./data/train_dataset_{HYPERPARAMS['dataset']}.pt",
        name=f"train_dataset_{HYPERPARAMS['dataset']}",
        type="dataset",
    )
    wandb.log_artifact(
        f"./data/valid_dataset_{HYPERPARAMS['dataset']}.pt",
        name=f"valid_dataset_{HYPERPARAMS['dataset']}",
        type="dataset",
    )
    wandb.log_artifact(
        f"./data/test_dataset_{HYPERPARAMS['dataset']}.pt",
        name=f"test_dataset_{HYPERPARAMS['dataset']}",
        type="dataset",
    )


def train_model():
    if HYPERPARAMS["wandb"]:
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="train-model",
        )
        train_artifact = wandb.use_artifact(
            f"train_dataset_{HYPERPARAMS['dataset']}" + ":latest"
        )
        train_dataset_dir = train_artifact.download()
        train_dataset = torch.load(
            os.path.join(
                train_dataset_dir, f"train_dataset_{HYPERPARAMS['dataset']}.pt"
            )
        )
        valid_artifact = wandb.use_artifact(
            f"valid_dataset_{HYPERPARAMS['dataset']}" + ":latest"
        )
        valid_dataset_dir = valid_artifact.download()
        valid_dataset = torch.load(
            os.path.join(
                valid_dataset_dir, f"valid_dataset_{HYPERPARAMS['dataset']}.pt"
            )
        )
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="train-model",
        )
        train_dataset = torch.load(f"./data/train_dataset_{HYPERPARAMS['dataset']}.pt")
        valid_dataset = torch.load(f"./data/valid_dataset_{HYPERPARAMS['dataset']}.pt")

    model = MODELS[wandb.config["model"]](wandb.config)
    wandb.watch(model)

    train(model, train_dataset, valid_dataset, wandb.config)


def evaluate_model():
    if HYPERPARAMS["wandb"]:
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="evaluate-model",
        )
        test_dataset = wandb.use_artifact("test_dataset" + ":latest")
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="evaluate-model",
        )
        test_dataset = torch.load("./data/test_dataset.pt")

    model_artifact = wandb.use_artifact(f"{wandb.config['model']}-best" + ":latest")
    model_dir = model_artifact.download()
    model = torch.load(os.path.join(model_dir, f"{wandb.config['model']}-best.pt"))
    wandb.watch(model)

    evaluate(model, test_dataset, wandb.config)


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
    if len(sys.argv) == 1:
        train_model()
    if sys.argv[1] == "sweep":
        create_sweep()
    elif sys.argv[1] == "agent":
        sweep_id = sys.argv[2]
        run_agent(sweep_id)
    elif sys.argv[1] == "data":
        generate_dataset()
    else:
        train_model()
