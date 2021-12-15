import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel


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
    def __init__(self, num_classes, layer_sizes=(768, 768, 768), kernel_size=10):
        super(CNNClassifier, self).__init__()
        self.layer_sizes = layer_sizes
        self.kernel_size = kernel_size
        self.embedding_layer = BertModel.from_pretrained("bert-base-cased")
        self.conv_blocks = nn.ModuleList(
            [
                self.create_conv_block(
                    in_f, out_f, kernel_size=self.kernel_size, padding="same"
                )
                for in_f, out_f in zip(self.layer_sizes, self.layer_sizes[1:])
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(layer_sizes[-1], 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, num_classes),
        )

    def forward(self, input_id, mask):
        mask = mask.float()
        x, _ = self.embedding_layer(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        x = x.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            # x *= mask

        pool_output = torch.sum(x, 2) / torch.sum(mask)

        output = self.output_layer(pool_output)

        return output

    def create_conv_block(self, in_features, out_features, *args, **kwargs):
        return nn.Sequential(
            nn.Conv1d(in_features, out_features, *args, **kwargs),
            nn.ReLU(inplace=False),
        )
