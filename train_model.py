import pandas as pd
from transformers import BertTokenizer
from classifiers import BertClassifier, CNNClassifier
from utils import train, evaluate, split_train_valid, data_to_df
from torchtext.datasets import AG_NEWS

HYPERPARAMS = {
    "dataset": "AG_NEWS",
    "num_classes": 4,
    "valid_prop": 0.15,
    "epochs": 5,
    "lr": 1e-4,
    "batch_size": 2,
    "model": "cnn",
    # cnn hyperparams
    "num_layers": 3,
    "hidden_dim": 256,
    "kernel_size": 10,
}

MODELS = {
    "cnn": lambda hyperparams: CNNClassifier(hyperparams),
    "transformer": lambda hyperparams: BertClassifier(hyperparams),
}

def train_model():
    train_iter, _ = AG_NEWS(root=".data", split=("train", "test"))

    train_iter, valid_iter = split_train_valid(train_iter, HYPERPARAMS["valid_prop"])

    train_df = data_to_df(train_iter)
    valid_df = data_to_df(valid_iter)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    model = MODELS[HYPERPARAMS["model"]](HYPERPARAMS)

    train(model, valid_df, valid_df, tokenizer, HYPERPARAMS)


def evaluate_model():
    _, test_iter = AG_NEWS(root=".data", split=("train", "test"))

    test_df = data_to_df(test_iter)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    model = MODELS[HYPERPARAMS["model"]](HYPERPARAMS)

    evaluate(model, test_df, tokenizer)

if __name__=='__main__':
    train_model()
