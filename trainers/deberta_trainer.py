import torch
import os
from pathlib import Path
from Data.reddit_dataset import RedditDataset
from models.deberta import custom_deberta
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import DebertaTokenizerFast, AdamW, get_linear_schedule_with_warmup
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

data_dir = os.path.join(Path(__file__).resolve().parents[1], "datasets")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class deberta_trainer():
    def __init__(self, model, train_dataset, test_dataset,epochs, batch_size, learning_rate, device, val_dataset=None):
        self.model = model
        self.ix_to_class = train_dataset.ix_to_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.load_tokenizer()
        self.create_data_loaders()
        self.create_optimizer()
    
    def collate_fn(self, batch):
        data, target = zip(*batch)
        output = self.tokenizer(data, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        target = torch.tensor(target)
        return output, target
    
    def load_tokenizer(self):
        self.tokenizer = DebertaTokenizerFast.from_pretrained(os.path.join(data_dir, "DebertaTokenizer"), do_lower_case=True, max_len=512, pad_to_max_length=True)
    
    def create_data_loaders(self):
        if self.val_dataset is None:
            train_set, valid_set  = random_split(self.train_dataset, [0.8, 0.2])
        else:
            train_set = self.train_dataset
            valid_set = self.val_dataset
        
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
    
    def create_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(self.train_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0.01*total_steps, num_training_steps=total_steps)
    
    
    def train(self):
        min_val_loss = 100000
        loss_total = 0
        loss_num = 0
        self.model = torch.compile(self.model)
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            for step, (data, target) in enumerate(self.train_loader):
                input_ids, attention_mask, token_type_ids, target = data['input_ids'].to(self.device), data['attention_mask'].to(self.device),data['token_type_ids'].to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
                loss_num += 1
                if step % 100 == 0 and step!=0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_total/loss_num}")
            
            print(f"Epoch: {epoch}, Loss: {loss_total/loss_num}")
            print("Evaluating on validation set")
            val_loss,val_acc, val_f1 = self.evaluation(data_set='val')
            
            print(f"Epoch: {epoch}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val F1: {val_f1}")
            if val_loss < min_val_loss:
                
                print("Saving model")
                torch.save(self.model.state_dict(), "deberta_checkpoint.pt")
                min_val_loss = val_loss
        
    
    def evaluation(self, data_set='val'):
        
        if data_set == 'val':
            data_loader = self.valid_loader
        elif data_set == 'test':
            data_loader = self.test_loader
            
            
        total_logits = []
        total_labels = []
        self.model.eval()
        loss_total = 0
        loss_num = 0
        
        for step, (data, target) in enumerate(data_loader):
            input_ids, attention_mask, token_type_ids, target = data['input_ids'].to(self.device), data['attention_mask'].to(self.device),data['token_type_ids'].to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, token_type_ids)
            
            total_logits.append(logits)
            loss = self.criterion(logits, target)
            total_labels += target.tolist()
            loss_total += loss.item()
            loss_num += 1
        
        total_loss = loss_total/loss_num
        total_logits = F.softmax(torch.cat(total_logits, dim=0).detach(), dim=1)
        total_labels = torch.tensor(total_labels, device=self.device)
        accuracy = multiclass_accuracy(total_logits, total_labels)
        f1_score = multiclass_f1_score(total_logits, total_labels)

        return total_loss, accuracy, f1_score
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

