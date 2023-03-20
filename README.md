# depression-detector
A deep learning approach to detect depression from social media text

## How to run training for fine-tuning Deberta model
1. Download the dataset, weights and tokenizer and place them in ```datasets``` folder
2. Navigate to ```trainers``` folder
3. Run ```python deberta_trainer.py```
This will run the training and evaluates the model and saves the model weights.

## How to run training for FastText + Transformer Encoder model
1. Download the dataset, embeddings and place them in ```datasets``` folder
2. Navigate to ```trainers``` folder
3. Run ```python transformers_trainer.py```
This will run the training and evaluates the model and saves the model weights.

## Requirements
1. pytorch=2.0.0 (This is needed as I used ```torch.compile()``` to optimise the computation graph)
2. Transformers
3. TorchEval (Note: please check if the code using this package is working. I didnt get to test it.)
