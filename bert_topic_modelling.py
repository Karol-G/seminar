from bertopic.model import BERTopic
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import json

#docs = fetch_20newsgroups(subset='all')['data']

data_path = "D:/Datasets/visuelle_trendanalyse/arxiv/arxiv-metadata-oai-snapshot.json"

data = []
with open(data_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

data = [entry['abstract'] for entry in data]

model = BERTopic("distilbert-base-nli-mean-tokens", verbose=True)
topics, probabilities = model.fit_transform(data, show_progress_bar=True)
model.save("arxiv_model")
print(model.get_topic(0))
