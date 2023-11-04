import argparse
import json
import tqdm
from models.extractive_summarizer import ExtractiveSummarizer
import cProfile

args = argparse.ArgumentParser()
args.add_argument('--train_data', type=str, default='data/train.greedy_sent.json')
args.add_argument('--eval_data', type=str, default='data/validation.json')
args.add_argument('--skip_vectors', action='store_true')
args.add_argument('--force_idf', action='store_true')
args.add_argument('--less_vectors', action='store_true')
args.add_argument('--less_articles', action='store_true')
args = args.parse_args()

model = ExtractiveSummarizer(args.skip_vectors, args.force_idf, args.less_vectors)

with open(args.train_data, 'r') as f:
    train_data = json.load(f)

train_articles = [article['article'] for article in train_data]
train_highligt_decisions = [article['greedy_n_best_indices'] for article in train_data]

preprocessed_train_articles = model.preprocess(train_articles)
SHORTENED_NUM = 100
if args.less_articles:
    model.train(preprocessed_train_articles[:SHORTENED_NUM], train_highligt_decisions[:SHORTENED_NUM])
else:
    model.train(preprocessed_train_articles, train_highligt_decisions)
with open(args.eval_data, 'r') as f:
    eval_data = json.load(f)

eval_articles = [article['article'] for article in eval_data]
preprocessed_eval_articles = model.preprocess(eval_articles)
summaries = model.predict(preprocessed_eval_articles)
eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]

with open("prediction_file.json", "w", encoding="utf-8") as f:
    json.dump(eval_out_data, fp=f, ensure_ascii=False, indent=4)