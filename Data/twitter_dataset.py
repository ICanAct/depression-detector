from torch.utils.data import Dataset
import os
import numpy as np
import torch
from pathlib import Path
import pandas as pd

data_dir = os.path.join(Path(__file__).resolve().parents[1], "datasets")


class TwitterDataset(Dataset):
    def __init__(self, file, bert=False) -> None:
        super().__init__()
        # define the data loading logic here
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.ix_to_class = {0: 'anxiety', 1: 'depression', 2: 'bipolar', 3: 'ptsd', 4: 'eatingdisorder'}
        self.frame = pd.read_csv(os.path.join(data_dir, file), usecols=['cleanedtweet', 'label'])
        self.frame = self.frame.dropna()
        self.samples = self.frame['cleanedtweet'].tolist()
        self.labels = self.frame['label'].tolist()
        self.bert = bert
        if not bert:
            self.load_tokenizer()
        
        assert len(self.samples) == len(self.labels)

    def __len__(self) -> int:
        # define the length of the dataset here
        return len(self.samples)

    def __getitem__(self, index):
        # define how to get a sample from the dataset here
        if not self.bert:
            sample_data = self.convert_text_to_input_ids(self.samples[index])
        else:
            sample_data = self.samples[index]
        
        sample_label = self.labels[index]
        return sample_data, sample_label

 
    def load_tokenizer(self):
        # this is for fasttext
        vocab = np.load(os.path.join(data_dir, "vocab_npa.npy"), allow_pickle=True)
        self.word2idx = {term:idx for idx,term in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}
        
    def convert_text_to_input_ids(self,text):
        words = text.strip().split()
        for i in range(len(words)):
            if words[i] not in self.word2idx:
                words[i] = self.word2idx[self.unk_token]
            else:
                words[i] = self.word2idx[words[i]]
        return torch.Tensor(words).long()