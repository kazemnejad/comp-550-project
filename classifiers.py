import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
import math


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5, num_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class CNNClassifier(nn.Module):
    def __init__(self, hyperparams):
        super(CNNClassifier, self).__init__()
        self.num_classes = hyperparams["num_classes"]
        self.layer_sizes = [hyperparams["embedding_size"]] + [
            hyperparams["hidden_dim"] for i in range(hyperparams["num_layers"] - 1)
        ]
        self.kernel_size = hyperparams["kernel_size"]
        self.embedding_layer = nn.Embedding(
            hyperparams["vocab_size"], hyperparams["embedding_size"]
        )
        self.conv_blocks = nn.ModuleList(
            [
                self.create_conv_block(
                    in_f, out_f, kernel_size=self.kernel_size, padding="same"
                )
                for in_f, out_f in zip(self.layer_sizes, self.layer_sizes[1:])
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.layer_sizes[-1], 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, self.num_classes),
        )

    def forward(self, inputs, mask):
        x = self.embedding_layer(inputs)
        x = x.permute(0, 2, 1)
        mask = mask[:, None, :]
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = x * mask

        pool_output = torch.sum(x, 2) / torch.sum(mask)

        output = self.output_layer(pool_output)

        return output

    def create_conv_block(self, in_features, out_features, *args, **kwargs):
        return nn.Sequential(
            nn.Conv1d(in_features, out_features, *args, **kwargs),
            nn.ReLU(inplace=False),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        pe = torch.zeros(max_len, 1, embedding_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, hyperparams):
        super(TransformerClassifier, self).__init__()
        self.num_classes = hyperparams["num_classes"]
        self.embedding_size = hyperparams["embedding_size"]
        self.num_heads = hyperparams["num_heads"]
        self.embedding_layer = nn.Embedding(
            hyperparams["vocab_size"], hyperparams["embedding_size"]
        )
        self.pos_encoder = PositionalEncoding(
            hyperparams["embedding_size"], hyperparams["max_len"]
        )
        encoder_layers = nn.TransformerEncoderLayer(
            hyperparams["embedding_size"],
            hyperparams["num_heads"],
            hyperparams["embedding_size"],
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, hyperparams["num_layers"]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hyperparams["embedding_size"], 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, self.num_classes),
        )

    def forward(self, inputs, mask):
        mask = mask.float()
        num_tokens = torch.sum(mask)
        mask = torch.einsum("ij,ik->ijk", mask, mask)
        mask = mask.repeat(self.num_heads, 1, 1)
        inputs = self.embedding_layer(inputs) * math.sqrt(self.embedding_size)
        inputs = self.pos_encoder(inputs)
        output = self.transformer_encoder(inputs, mask)

        pool_output = torch.sum(output, 1) / num_tokens

        output = self.output_layer(pool_output)

        return output


class LSTMClassifier(nn.Module):
    def __init__(self, hyperparams):
        super(LSTMClassifier, self).__init__()
        self.num_classes = hyperparams["num_classes"]
        self.layer_sizes = [hyperparams["embedding_size"]] + [
            hyperparams["hidden_dim"] for i in range(hyperparams["num_layers"] - 1)
        ]
        self.kernel_size = hyperparams["kernel_size"]
        self.embedding_layer = nn.Embedding(
            hyperparams["vocab_size"], hyperparams["embedding_size"]
        )
        self.lstm = nn.LSTM(
            hyperparams["embedding_size"],
            hyperparams["hidden_dim"],
            batch_first=True,
            bidirectional=True,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hyperparams["hidden_dim"]*2, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, self.num_classes),
        )

    def forward(self, inputs, _):
        x = self.embedding_layer(inputs)
        # x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        output = self.output_layer(x[:, -1, :])

        return output
