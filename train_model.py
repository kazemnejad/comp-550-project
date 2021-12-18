import numpy as np
from classifiers import BertClassifier, CNNClassifier
from utils import train, evaluate, split_train_valid, get_dataset
import torch
from torchtext.datasets import AG_NEWS
import wandb
import os
import sys

HYPERPARAMS = {
    "dataset": "AG_NEWS",
    "embeddings": "bert",
    "num_classes": 4,
    "valid_prop": 0.15,
    "epochs": 5,
    "lr": 1e-4,
    "weight_decay": 1e-10,
    "batch_size": 32,
    "vocab_size": 100000,
    "embedding_size": 32,
    "model": "cnn",
    "seed": 0,
    "wandb": True,
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
        "embedding_size": {"values": [32, 64, 128]},
    },
}


MODELS = {
    "cnn": lambda hyperparams: CNNClassifier(hyperparams),
    "transformer": lambda hyperparams: BertClassifier(hyperparams),
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

    train_iter, test_iter = AG_NEWS(root=".data", split=("train", "test"))
    train_iter, valid_iter = split_train_valid(train_iter, HYPERPARAMS["valid_prop"])

    train_dataset, valid_dataset, test_dataset = get_dataset(
        train_iter, valid_iter, test_iter
    )

    torch.save(train_dataset, "./data/train_dataset.pt")
    torch.save(valid_dataset, "./data/valid_dataset.pt")
    torch.save(test_dataset, "./data/test_dataset.pt")

    wandb.log_artifact("./data/train_dataset.pt", name="train_dataset", type="dataset")
    wandb.log_artifact("./data/valid_dataset.pt", name="valid_dataset", type="dataset")
    wandb.log_artifact("./data/test_dataset.pt", name="test_dataset", type="dataset")


def train_model():
    if HYPERPARAMS["wandb"]:
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="train-model",
        )
        train_dataset = wandb.use_artifact("train_dataset" + ":latest")
        valid_dataset = wandb.use_artifact("valid_dataset" + ":latest")
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(
            config=HYPERPARAMS,
            project="classification_models",
            entity="comp-555-project",
            job_type="train-model",
        )
        train_dataset = torch.load("./data/train_dataset.pt")
        valid_dataset = torch.load("./data/valid_dataset.pt")

    model = MODELS[HYPERPARAMS["model"]](HYPERPARAMS)
    wandb.watch(model)

    train(model, train_dataset, valid_dataset, HYPERPARAMS)


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

    model = wandb.use_artifact(f"{HYPERPARAMS['model']}-best" + ":latest")
    wandb.watch(model)

    evaluate(model, test_dataset, HYPERPARAMS)


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
