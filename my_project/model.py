from typing import Dict

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from torch import nn

@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(  # type: ignore
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

@Model.register("lyr_cnn_classifier")
class CNNClassifier(Model):
    def __init__(
        self, 
        vocab: Vocabulary, 
        embedder: TextFieldEmbedder, 
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
    ):
        super().__init__(vocab)
        self.embedder = embedder
        
        self.layer_sizes = [embedding_dim] + [
            hidden_dim for i in range(num_layers - 1)
        ]

        self.kernel_size = kernel_size
        self.conv_blocks = nn.ModuleList(
            [
                self.create_conv_block(
                    in_f, out_f, kernel_size=self.kernel_size, padding="same"
                )
                for in_f, out_f in zip(self.layer_sizes, self.layer_sizes[1:])
            ]
        )
        

        num_labels = vocab.get_vocab_size("labels")
        self.classifier = nn.Sequential(
            nn.Linear(self.layer_sizes[-1], 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, num_labels),
        )
        self.accuracy = CategoricalAccuracy()

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        text = tokens
        x = self.embedder(text)
        
        mask = util.get_text_field_mask(text)
        mask = mask[:, None, :]
        x = x.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = x * mask

        pool_output = torch.sum(x, 2) / torch.sum(mask)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(pool_output)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def create_conv_block(self, in_features, out_features, *args, **kwargs):
        return nn.Sequential(
            nn.Conv1d(in_features, out_features, *args, **kwargs),
            nn.ReLU(inplace=False),
        )
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}