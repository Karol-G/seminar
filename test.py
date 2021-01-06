from sentence_transformers import SentenceTransformer
from sentence_transformers import models

model_path = "/gris/gris-f/homelv/kgotkows/datasets/arxiv/arxiv_trained_embedding/model/"

# model = SentenceTransformer(model_path)

# Use BERT for mapping tokens to embeddings
word_embedding_model = models.DistilBERT(model_path)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])