# depression-detector
A machine learning approach to detect mental health issues from social media content from Reddit and Twitter.
## Data scraping
We have scrapped data from Twitter and reddit using the following notebooks.
- Twitter data scraping is done using ```notebooks/Data Scraping Twitter.ipynb```
- Reddit data scraping is done using ```notebooks/Data Scraping Reddit.ipynb```
- We have also scraped neutral data from Reddit using ```notebooks/Data Scraping Reddit Neutral.ipynb```

## Data preprocessing
We have implemented extensive preprocessing steps to clean the data and remove noise. 
- Twitter data preprocessing is done using ```notebooks/clean_and_export_tweets.ipynb```
- Reddit data preprocessing is done using ```notebooks/clean_and_export_reddit.ipynb```
- The preprocessing steps are implemented in ```preprocess_tweets.py```

## Model Training
We have multiple models for this task. 
- Traditional Models (Logistic Regression, KNN): ```notebooks/ClassicalMachineLearning.ipynb```
- Basline Model: FastText + Multi Layer Perceptron: ```notebooks/Baseline.ipynb```
- Custom MLP model: ```notebooks/MLP.ipynb```
- CNN model: ```notebooks/CNN.ipynb```
- RNN model: ```notebooks/RNN.ipynb```
- Transformer model.
- Deberta model.

### How to run training for FastText + Transformer Encoder model
- Download the dataset, embeddings and place them in ```datasets``` folder
- Run ```python main_transformer_run.py```
This will run the training and evaluates the model and saves the model weights.
### Pre-training DeBERTa model
- Download the dataset and place it in ```datasets``` folder
- Run ```python train_tokenizer.py```. This trains the tokenizer required for Deberta model.
- Run ```python train_deberta.py```. This trains the Deberta model.

### How to run training for fine-tuning Deberta model
- Download the dataset, weights and tokenizer and place them in ```datasets``` folder
- Run ```python main_deberta_run.py```
This will run the training and evaluates the model and saves the model weights.


### Evaluating the Transformer and DeBERTa on test set
- Download the dataset, embeddings and model weights and place them in ```datasets``` folder.
- For transformers, run ```python transformer_inference.py```
- For Deberta, run ```python deberta_inference.py```
