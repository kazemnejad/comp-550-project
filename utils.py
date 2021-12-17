import torch
from torch import nn
from torch.optim import Adam
from torch.optim import Adam
from torch.utils.data.dataset import random_split
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


def data_to_df(data_iter):
    data = [[sample[1], sample[0]] for sample in data_iter]

    data_df = pd.DataFrame(data)
    data_df.columns = ["text", "labels"]

    return data_df


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):

        self.labels = [label for label in df["labels"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

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


def train(model, embedding_model, train_data, valid_data, tokenizer, hyperparams):
    train_dataset = Dataset(train_data, tokenizer)
    valid_dataset = Dataset(valid_data, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=hyperparams["lr"])

    if use_cuda:
        model = model.cuda()
        embedding_model = embedding_model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(hyperparams["epochs"]):
        total_acc_train = 0
        total_loss_train = 0
        total_acc_valid = 0
        total_loss_valid = 0
        best_valid_acc = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device).long()
            train_label = train_label - 1
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            embeddings, _ = embedding_model(
                input_ids=input_id, attention_mask=mask, return_dict=False
            )
            with profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with record_function("forward"):
                    output = model(embeddings, mask)

                with record_function("loss"):
                    batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                with record_function("backward"):
                    batch_loss.backward()
                optimizer.step()
            print(
                prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=25)
            )
            print("\n")
            print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=25))

        wandb.log(
            {
                "Train loss": total_loss_train / len(train_data),
                "Train accuracy": total_acc_train / len(train_data),
            },
            commit=False,
        )

        with torch.no_grad():
            for valid_input, valid_label in tqdm(valid_dataloader):
                valid_label = valid_label.to(device).long()
                valid_label = valid_label - 1
                mask = valid_input["attention_mask"].to(device)
                input_id = valid_input["input_ids"].squeeze(1).to(device)

                embeddings, _ = embedding_model(
                    input_ids=input_id, attention_mask=mask, return_dict=False
                )
                output = model(embeddings, mask)

                batch_loss = criterion(output, valid_label)
                total_loss_valid += batch_loss.item()

                acc = (output.argmax(dim=1) == valid_label).sum().item()
                total_acc_valid += acc

        wandb.log(
            {
                "Valid loss": total_loss_valid / len(valid_data),
                "Valid accuracy": total_acc_valid / len(valid_data),
            }
        )

        if total_acc_valid / len(valid_data) > best_valid_acc:
            wandb.run.summary["Best valid accuracy"] = total_acc_valid / len(valid_data)
            best_valid_acc = total_acc_valid / len(valid_data)

            torch.save(model.state_dict(), f"./models/{hyperparams['model']}-best.pt")
            wandb.log_artifact(
                f"./models/{hyperparams['model']}-best.pt",
                name=f"{hyperparams['model']}-best",
                type="model",
            )

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f}\
                | Valid Accuracy: {total_acc_valid / len(valid_data): .3f}"
        )


def evaluate(model, test_data, tokenizer):
    test = Dataset(test_data, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")

