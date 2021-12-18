from itertools import chain
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb

from torch.profiler import profile, record_function, ProfilerActivity

torch.autograd.set_detect_anomaly(True)


def split_train_valid(train_iter, valid_prop):
    train_data = list(train_iter)
    valid_len = int(len(train_data) * valid_prop)
    split_lenghts = [
        len(train_data) - valid_len,
        valid_len,
    ]
    train_iter, valid_iter = random_split(train_data, split_lenghts)

    return train_iter, valid_iter


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_iter, label_pipeline, text_pipeline):
        self.labels = []
        self.texts = []
        for label, text in data_iter:
            self.labels.append(label_pipeline(label))
            self.texts.append(text_pipeline(text))

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def get_dataset(train_iter, valid_iter, test_iter, sample_len=512):
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    full_data_iter = chain(train_iter, valid_iter, test_iter)
    vocab = build_vocab_from_iterator(yield_tokens(full_data_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(text):
        index_list = vocab(tokenizer(text))
        mask = torch.tensor(
            [1] * len(index_list) + [0] * (sample_len - len(index_list))
        )
        text_sample = [index + 1 for index in vocab(tokenizer(text))] + [0] * (
            sample_len - len(index_list)
        )
        text_sample = torch.tensor(text_sample)

        return text_sample, mask

    label_pipeline = lambda x: torch.tensor(int(x) - 1)

    train_dataset = Dataset(train_iter, label_pipeline, text_pipeline)
    valid_dataset = Dataset(valid_iter, label_pipeline, text_pipeline)
    test_dataset = Dataset(train_iter, label_pipeline, text_pipeline)

    return train_dataset, valid_dataset, test_dataset


def train(model, train_dataset, valid_dataset, hyperparams):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
    )

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    wandb.run.summary["Number parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    for epoch_num in range(hyperparams["epochs"]):
        total_acc_train = 0
        total_loss_train = 0
        total_acc_valid = 0
        total_loss_valid = 0
        best_valid_acc = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device).long()
            input_text, mask = train_input
            mask = mask.to(device)
            input_text = input_text.to(device)

            output = model(input_text, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        wandb.log(
            {
                "Train loss": total_loss_train / len(train_dataset),
                "Train accuracy": total_acc_train / len(train_dataset),
            },
            commit=False,
        )

        with torch.no_grad():
            for valid_input, valid_label in tqdm(valid_dataloader):
                valid_label = valid_label.to(device).long()
                input_text, mask = valid_input
                mask = mask.to(device)
                input_text = input_text.to(device)

                output = model(input_text, mask)

                batch_loss = criterion(output, valid_label)
                total_loss_valid += batch_loss.item()

                acc = (output.argmax(dim=1) == valid_label).sum().item()
                total_acc_valid += acc

        wandb.log(
            {
                "Valid loss": total_loss_valid / len(valid_dataset),
                "Valid accuracy": total_acc_valid / len(valid_dataset),
            }
        )

        if total_acc_valid / len(valid_dataset) > best_valid_acc:
            wandb.run.summary["Best valid accuracy"] = total_acc_valid / len(
                valid_dataset
            )
            best_valid_acc = total_acc_valid / len(valid_dataset)

            torch.save(model.state_dict(), f"./models/{hyperparams['model']}-best.pt")
            wandb.log_artifact(
                f"./models/{hyperparams['model']}-best.pt",
                name=f"{hyperparams['model']}-best",
                type="model",
            )

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataset): .3f}\
                | Valid Accuracy: {total_acc_valid / len(valid_dataset): .3f}"
        )


def evaluate(model, test_dataset, hyperparams):
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"]
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device).long()
            input_text, mask = test_input
            mask = mask.to(device)
            input_text = input_text.to(device)

            output = model(input_text, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_dataset): .3f}")

