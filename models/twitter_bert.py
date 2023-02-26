from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class BerTweet(nn.Module):
    
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
         
        self.basemodel = AutoModel.from_pretrained("vinai/bertweet-base")
        self.drop = nn.Dropout(0.3)
        # edit this layer according to your task
        self.out = nn.Linear(768, 1)
        
    def forward(self, ids, mask, token_type_ids):
        x = self.basemodel(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.drop(x[0])
        x = self.out(x)
        x = F.relu(x)
        
        return x