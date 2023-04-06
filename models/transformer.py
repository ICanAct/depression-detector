import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import math
from pathlib import Path

data_dir = os.path.join(Path(__file__).resolve().parents[1], "datasets")

class custom_transformer(torch.nn.Module):
    def __init__(self, num_classes, dropout, custom_embeddings=True):
        super(custom_transformer, self).__init__()
        self.custom_embeddings = custom_embeddings
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.pos_encoder = PositionalEncoding(300, dropout=dropout)
        
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.custom_embeddings:
            self.load_embeddings()
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embeddings).float(), freeze=True)
        
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=300, nhead=3, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        
        self.output_projection = torch.nn.Linear(300, 64)
        self.classifier_head = torch.nn.Linear(64, num_classes)
        
        self.init_weights()

    def load_embeddings(self):
        # load the embeddings here
        self.embeddings = np.load(os.path.join(data_dir, "embs_npa.npy"), allow_pickle=True)
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.classifier_head.bias.data.zero_()
        self.classifier_head.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x, src_padding_mask):
        if self.custom_embeddings:
            x = self.embedding(x)
        
        # add pos encoding
        x = x * math.sqrt(300)
        x = self.pos_encoder(x)
        # transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_padding_mask)
        x = torch.mean(x, dim=0)
        x = self.output_projection(x)
        x = F.relu(x)
        x = self.classifier_head(x)
        return x

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)