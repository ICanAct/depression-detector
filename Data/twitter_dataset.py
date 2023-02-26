from torch.utils.data import Dataset
from typing import List, _T_co


# Class: TwitterDataset
class TwitterDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # define the data loading logic here
    
    
    def __len__(self) -> int:
        # define the length of the dataset here
        return 0
    
    def __getitem__(self, index: int) -> _T_co:
        return super().__getitem__(index)
    
    def _tokenize(self, text: str) -> List[str]:
        # define the tokenization logic here
        return []