import torch
from Data.reddit_dataset import RedditDataset
from models.transformer import custom_transformer
from trainers.transformers_trainer import custom_transformers_trainer

if __name__ == '__main__':
    
    DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = RedditDataset('train.csv', bert=False)
    test_data = RedditDataset('reddit_golden_test.csv', bert=False)
    valid_data = RedditDataset('test.csv', bert=False)
    model = custom_transformer(num_classes=3, dropout=0.5, custom_embeddings=True)
    trainer = custom_transformers_trainer(model, train_data, test_data, epochs=30, batch_size=32, learning_rate=0.001, device=DEVICE)
    trainer.train()
    print("Evaluating on test set")
    trainer.evaluation('test')
    # path to save the weights. 
    #trainer.save_model('transformer_weights.pt')
