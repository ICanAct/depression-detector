import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

data_dir = os.path.join(Path(__file__).resolve().parents[1], "datasets")

class custom_transformer(torch.nn.Module):
    def __init__(self, num_classes, dropout, custom_embeddings=True):
        super(custom_transformer, self).__init__()
        self.custom_embeddings = custom_embeddings
        self.num_classes = num_classes
        self.dropout = dropout
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.custom_embeddings:
            self.load_embeddings()
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embeddings).float(), freeze=True)
        
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=300, nhead=3, batch_first=True, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        
        self.output_projection = torch.nn.Linear(300, 64)
        self.classifier_head = torch.nn.Linear(64, num_classes)

    def load_embeddings(self):
        # load the embeddings here
        self.embeddings = np.load(os.path.join(data_dir, "embs_npa.npy"), allow_pickle=True)
        
    def forward(self, x, src_padding_mask):
        if self.custom_embeddings:
            x = self.embedding(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=src_padding_mask)
        x = torch.mean(x, dim=1)
        x = self.output_projection(x)
        x = F.relu(x)
        x = self.classifier_head(x)
        return x
