import torch
from Data.reddit_dataset import RedditDataset
from models.deberta import custom_deberta
from trainers.deberta_trainer import deberta_trainer
if __name__ == "__main__":
    
    DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = RedditDataset('train.csv', bert=True)
    test_data = RedditDataset('reddit_golden_test.csv', bert=True)
    valid_data = RedditDataset('test.csv', bert=True)
    model = custom_deberta(num_classes=3, dropout=0.3)
    trainer = deberta_trainer(model, train_data, test_data, epochs=10, batch_size=512, learning_rate=1e-5, device=DEVICE, val_dataset=valid_data)
    trainer.train()
    print("Evaluating on test set")
    test_loss, test_acc, test_f1 = trainer.evaluation('test')
    
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}, Test F1: {test_f1}")
    
    # path to save the weights. 
    trainer.save_model('deberta_model.pt')
