import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from transformers import DebertaModel

data_dir = os.path.join(Path(__file__).resolve().parents[1], "datasets")

class custom_deberta(torch.nn.Module):
    def __init__(self, num_classes, dropout):
        super(custom_deberta, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.load_deberta()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.output_projection = torch.nn.Linear(self.deberta.config.hidden_size, 32)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.classifier = torch.nn.Linear(32, self.num_classes)
        
    def load_deberta(self):
        self.deberta = DebertaModel.from_pretrained(os.path.join(data_dir, "DebertaLatestCheckpoint"))
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.deberta(input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
        output = output[0]
        #output = self.dropout_layer(output)
        # This is to get a sense of the whole sentence embedding. (because we are using a classification task)
        output = torch.mean(output, dim=1)
        output = F.relu(self.output_projection(output))
        output = self.classifier(output)
        return output
    

    