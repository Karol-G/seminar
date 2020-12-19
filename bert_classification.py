import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import MultiLabelClassificationModel

train_path = "D:/Datasets/visuelle_trendanalyse/toxic _comment _classification/train.csv"
df = pd.read_csv(train_path)

def preprocess(df):
    df = df.rename(columns={"comment_text": "text"})
    df['labels'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.tolist()
    df = df[['text', 'labels']]
    return df

train_df, eval_df = train_test_split(df, test_size=0.2)
train_df = preprocess(train_df)
eval_df = preprocess(eval_df)
#
# model = MultiLabelClassificationModel(
#     'roberta',
#     'roberta-base',
#     num_labels=6,
#     args={
#         'train_batch_size':2,
#         'gradient_accumulation_steps':16,
#         'learning_rate': 3e-5,
#         'num_train_epochs': 3,
#         'max_seq_length': 512})
#
# model.train_model(train_df)