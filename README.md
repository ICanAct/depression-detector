# depression-detector
A machine learning approach to detect mental health issues from social media content from Reddit and Twitter.
## Data scraping
We have scrapped data from Twitter and reddit using the following notebooks.
1. Twitter data scraping is done using ```notebooks/Data Scraping Twitter.ipynb```
2. Reddit data scraping is done using ```notebooks/Data Scraping Reddit.ipynb```
3. We have also scraped neutral data from Reddit using ```notebooks/Data Scraping Reddit Neutral.ipynb```

## Data preprocessing
We have implemented extensive preprocessing steps to clean the data and remove noise. 
1. Twitter data preprocessing is done using ```notebooks/clean_and_export_tweets.ipynb```
2. Reddit data preprocessing is done using ```notebooks/clean_and_export_reddit.ipynb```
3. The preprocessing steps are implemented in ```preprocess_tweets.py```

## Model Training
We have multiple models for this task. 
- Traditional Models (Logistic Regression, KNN): ```notebooks/ClassicalMachineLearning.ipynb```
- Basline Model: FastText + Multi Layer Perceptron: ```notebooks/Baseline.ipynb```
- Custom MLP model: ```notebooks/MLP.ipynb```
- CNN model: ```notebooks/CNN.ipynb```
- RNN model: ```notebooks/RNN.ipynb```
- Transformer model.
- Deberta model.

### Pre-training DeBERTa model
1. Download the dataset and place it in ```datasets``` folder
2. Run 
### How to run training for fine-tuning Deberta model
1. Download the dataset, weights and tokenizer and place them in ```datasets``` folder
2. Run ```python main_deberta_run.py```
This will run the training and evaluates the model and saves the model weights.

## How to run training for FastText + Transformer Encoder model
1. Download the dataset, embeddings and place them in ```datasets``` folder
2. Run ```python main_transformer_run.py```
This will run the training and evaluates the model and saves the model weights.
## Evaluating the models on test set
1. Download the dataset, embeddings and model weights and place them in ```datasets``` folder.
2. For transformers, run ```python transformer_inference.py```
3. For Deberta, run ```python deberta_inference.py```
