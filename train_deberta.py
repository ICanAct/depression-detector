from itertools import chain
from datasets import load_dataset
from transformers import (
    DebertaTokenizerFast, 
    DataCollatorForLanguageModeling, 
    DebertaForMaskedLM, 
    DebertaConfig, 
    TrainingArguments, Trainer)

# remember to change the path
DATA_PATH = '/Users/krishnavyas/Downloads/sample_twitter_data.txt'
TOKENIZER_PATH = '/Users/krishnavyas/Desktop/NUS/Text Mining/project/depression-detector/dberta-tokenizer'
MODEL_SAVE_PATH = "<ADD PATH HERE>"

dataset = load_dataset("text", data_files=DATA_PATH, split='train')
# adjust the test size accordingly
d = dataset.train_test_split(test_size=0.1)

# load the tokenizer
tokenizer = DebertaTokenizerFast.from_pretrained(TOKENIZER_PATH)

# Check 512. Tweets are usually short
def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"], truncation=True, padding="max_length",
                   max_length=512, return_special_tokens_mask=True)

# tokenize the whole dataset
# tokenizing the train dataset
train_dataset = d["train"].map(encode_with_truncation, batched=True)
# tokenizing the testing dataset (eval set in our usecase)
test_dataset = d["test"].map(encode_with_truncation, batched=True)

train_dataset.set_format(type='torch' ,columns=["input_ids", "attention_mask", "special_tokens_mask"])
test_dataset.set_format(type='torch' ,columns=["input_ids", "attention_mask", "special_tokens_mask"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

# set the config. Think of how we can adjust the hidden size, num layers and heads 
# based on our input data characteristics.

config = DebertaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=4,
    hidden_act="gelu_new",
    intermediate_size=1024
)
model = DebertaForMaskedLM(config=config)
# training arguments
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=25,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=16, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()