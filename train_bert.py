from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


keyword = "title"
data_path = "/gris/gris-f/homelv/kgotkows/datasets/arxiv/arxiv_trained_embedding/arxiv_title.txt"
data_save_path = "/gris/gris-f/homelv/kgotkows/datasets/arxiv/arxiv_trained_embedding/"
print("data_save_path: ", data_save_path)

with open("/gris/gris-f/homelv/kgotkows/datasets/arxiv/title3/arxiv_title.pkl", 'rb') as handle:
    data = pickle.load(handle)

train, test = train_test_split(data, test_size=0.2)

with open(data_save_path + 'arxiv_title_train.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)

with open(data_save_path + 'arxiv_title_val.txt', 'w') as f:
    for item in test:
        f.write("%s\n" % item)

# # Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()
#
# # Customize training
# tokenizer.train(files=data_path, vocab_size=52_000, min_frequency=2, special_tokens=[
#     "<s>",
#     "<pad>",
#     "</s>",
#     "<unk>",
#     "<mask>",
# ])
#
# tokenizer.save_model(data_save_path + "tokenized")

# tokenizer = ByteLevelBPETokenizer(
#     data_save_path + "tokenized/vocab.json",
#     data_save_path + "tokenized/merges.txt",
# )
#
# config = RobertaConfig(
#     vocab_size=52_000,
#     max_position_embeddings=514,
#     num_attention_heads=12,
#     num_hidden_layers=6,
#     type_vocab_size=1,
# )
#
# tokenizer = RobertaTokenizerFast.from_pretrained(data_save_path + "tokenized", max_len=512)
#
# model = RobertaForMaskedLM(config=config).from_pretrained("roberta-base")  # roberta-base, distilbert-base-nli-stsb-mean-tokens, distilbert-base-uncased
#
# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path=data_path,
#     block_size=128,
# )
#
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )
#
# training_args = TrainingArguments(
#     output_dir=data_save_path + "model",
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_gpu_train_batch_size=64,
#     save_steps=10_000,
#     save_total_limit=2,
#     prediction_loss_only=True
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset,
# )
#
# trainer.train()