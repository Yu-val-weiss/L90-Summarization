import argparse
import json
from models.abstractive_summarizer import AbstractiveSummarizer

args = argparse.ArgumentParser()
args.add_argument('--train_data', type=str, default='data/train.json')
args.add_argument('--validation_data', type=str, default='data/validation.json')
args.add_argument('--eval_data', type=str, default='data/test.json')
args = args.parse_args()

model = AbstractiveSummarizer()

with open(args.train_data, 'r') as f:
    train_data = json.load(f)

with open(args.validation_data, 'r') as f:
    validation_data = json.load(f)

train_articles = [article['article'] for article in train_data]
train_summaries = [article['summary'] for article in train_data]

val_articles = [article['article'] for article in validation_data]
val_summaries = [article['summary'] for article in validation_data]

model.train(train_articles, train_summaries, val_articles, val_summaries)

with open(args.eval_data, 'r') as f:
    eval_data = json.load(f)

eval_articles = [article['article'] for article in eval_data]
summaries = model.predict(eval_articles)
eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]

print(json.dumps(eval_out_data, indent=4))