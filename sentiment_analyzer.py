from transformers import pipeline
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset

inputFile = "neutral.txt"
classifier = pipeline(task="sentiment-analysis", device=0)
dataset = load_dataset("text", data_files=inputFile)["train"]
for ind, prediction in enumerate(classifier(KeyDataset(dataset, "text"), batch_size=1024, truncation=True)):
	if prediction["label"] == "POSITIVE":
		print(dataset[ind]["text"])
