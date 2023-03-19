
import torch
from Data.reddit_dataset import RedditDataset
from models.transformer import custom_transformer
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score

class custom_transformers_trainer():
    def __init__(self, model, train_dataset, test_dataset,epochs, batch_size, learning_rate, device):
        self.model = model
        self.ix_to_class = train_dataset.ix_to_class
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.create_data_loaders()
        self.create_optimizer()
        
    def collate_fn(self, batch):
        data, target = zip(*batch)
        data = pad_sequence(data, batch_first=True, padding_value=0)
        target = torch.tensor(target)
        return data, target
    
    def create_data_loaders(self):
        train_set, valid_set  = random_split(self.train_dataset, [0.8, 0.2])
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self):
        
        loss_total = 0
        loss_num = 0
        model = torch.compile(self.model)
        model = model.to(self.device)
        model.train()
        for epoch in range(self.epochs):
            for step, (data, target) in enumerate(self.train_loader):
                # creating mask
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                src_mask = torch.zeros((data.shape[1], data.shape[1]), device=self.device).type(torch.bool)
                src_padding_mask = (data == 0)
                src_padding_mask = src_padding_mask.to(self.device)
                output = self.model(data, src_mask, src_padding_mask)
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
            self.evaluation(data_set='val')
            
    
    
    def evaluation(self, data_set='val'):
        
        if data_set == 'val':
            data_loader = self.valid_loader
        elif data_set == 'test':
            data_loader = self.test_loader
            
        model = torch.compile(self.model)
        model = model.to(self.device)
        total_logits = []
        total_labels = []
        model.eval()
        loss_total = 0
        loss_num = 0
        
        for data, targets in data_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            src_mask = torch.zeros((data.shape[1], data.shape[1]), device=self.device).type(torch.bool)
            src_padding_mask = (data == 0)
            src_padding_mask = src_padding_mask.to(self.device)
            logits = self.model(data, src_mask, src_padding_mask)
            total_logits.append(logits)
            loss = self.criterion(logits, targets)
            total_labels += targets.tolist()
            loss_total += loss.item()
            loss_num += 1
        
        print(f'Loss: {loss_total/loss_num}')
        
        _, total_logits = F.softmax(torch.cat(total_logits, dim=0).detach(), dim=1)
        total_labels = torch.tensor(total_labels)
        
    
        
        accuracy_obj = MulticlassAccuracy()
        accuracy_obj.update([total_logits, total_labels])
        accuracy = accuracy_obj.compute()
        print(f"Accuracy: {accuracy.item()}")
        
        f1_obj = MulticlassF1Score()
        f1_obj.update([total_logits, total_labels])
        f1_score = f1_obj.compute()
        
        print(f"F1 Score: {f1_score.item()}")
    

if __name__ == '__main__':
    
    DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = RedditDataset('train_reddit.csv', bert=False)
    test_data = RedditDataset('reddit_test.csv', bert=False)
    model = custom_transformer(num_classes=5, dropout=0.2, custom_embeddings=True)
    trainer = custom_transformers_trainer(model, train_data, test_data, epochs=30, batch_size=32, learning_rate=0.001, device=DEVICE)
    trainer.train()