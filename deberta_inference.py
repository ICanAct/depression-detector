import torch
from Data.reddit_dataset import RedditDataset
from Data.twitter_dataset import TwitterDataset
from models.deberta import custom_deberta
from trainers.deberta_trainer import deberta_trainer
import os
from pathlib import Path
data_dir = os.path.join(Path(__file__).resolve().parent, "datasets")

if __name__ == '__main__':
    
    DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = RedditDataset('train.csv', bert=True)
    test_data = RedditDataset('reddit_golden_test.csv', bert=True)
    valid_data = RedditDataset('test.csv', bert=True)
    twitter_data = TwitterDataset('twitter_anxiety_depression_neutral.csv', bert=True)
    model = custom_deberta(num_classes=3, dropout=0.1)
    checkpoint = torch.load("deberta_model.pt", map_location=DEVICE)
    
    # this logic is needed coz model was saved on gpu and we are loading it on cpu
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[10:]
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    
    trainer = deberta_trainer(model, train_data, test_data, epochs=20, batch_size=64, learning_rate=0.001, device=DEVICE, val_dataset=valid_data)    
    print("Evaluating on val set")
    test_loss, test_acc, test_f1 = trainer.evaluation('val')
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}, Test F1: {test_f1}")
    
    print("Evaluating on test set")
    test_loss, test_acc, test_f1 = trainer.evaluation('test')
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}, Test F1: {test_f1}")