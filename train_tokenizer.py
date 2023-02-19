import os
from tokenizers import models, Tokenizer, trainers, pre_tokenizers, decoders, processors, normalizers
from datasets import load_dataset


DATA_PATH = "/Users/krishnavyas/Desktop/NUS/Text Mining/project/depression-detector/tweets.txt"

dataset = load_dataset("text", data_files=DATA_PATH, split='train')

# batch iterator for iterating over the dataset
def batch_iterator(batch_size=10000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]
        
tokenizer = Tokenizer(models.BPE())
# load data using the datasets library
# Add normalizer to the tokenizer
# think about if we want it to be lowercase or not
tokenizer.normalizer = normalizers.BertNormalizer(clean_text=True, 
                                                  handle_chinese_chars=True, 
                                                  strip_accents=False, 
                                                  lowercase=False)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# These are same as the ones in the original BERT paper
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
# Think how much vocab size we want
trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=30000, min_frequency=5)
tokenizer.model = models.BPE()
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
# Save the tokenizer
dir_name = "dberta-tokenizer"

if os.path.isdir(dir_name) == False:
    os.mkdir(dir_name)  

tokenizer.model.save(dir_name)