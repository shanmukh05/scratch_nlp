import torch
import torch.nn as nn
import torch.nn.functional as F

from core.transformer.model import EncoderLayer, PositionalEncoding


class BERTModel(nn.Module):
    def __init__(self, config_dict):
        super(BERTModel, self).__init__()

        embed_dim = config_dict["model"]["d_model"]
        num_vocab = config_dict["dataset"]["num_vocab"]
        num_layers = config_dict["model"]["num_layers"]
        dropout = config_dict["model"]["dropout"]

        self.embed_layer = nn.Embedding(num_vocab, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.positional_encoding = PositionalEncoding(config_dict)
 
        self.encoder_layers = [EncoderLayer(config_dict) for _ in range(num_layers)]
        self.classifier_layer = nn.LazyLinear(num_vocab)

        self.nsp_classifier_layer = nn.LazyLinear(1)

    def forward(self, tokens):
        tokens_embed = self.dropout(self.positional_encoding(self.embed_layer(tokens)))

        enc_output = tokens_embed
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        output = self.classifier_layer(enc_output)
        output = nn.Softmax(dim=-1)(output)

        nsp_output = nn.Sigmoid()(self.nsp_classifier_layer(output[:, 0, :]))
    
        return output, nsp_output