import argparse
from evaluation.rouge_evaluator import RougeEvaluator
import json
import tqdm

args = argparse.ArgumentParser()
args.add_argument('--pred_data', type=str, default='prediction_file.json')
args.add_argument('--eval_data', type=str, default='data/validation.json')
args = args.parse_args()

evaluator = RougeEvaluator()

with open(args.eval_data, 'r') as f:
    eval_data = json.load(f)

with open(args.pred_data, 'r') as f:
    pred_data = json.load(f)

assert len(eval_data) == len(pred_data)

pred_sums = []
eval_sums = []
for eval, pred in tqdm.tqdm(zip(eval_data, pred_data), total=len(eval_data)):
    pred_sums.append(pred['summary'])
    eval_sums.append(eval['summary'])

scores = evaluator.batch_score(pred_sums, eval_sums)

file_path = "models/weights.json"
try:
    with open(file_path, "r") as file:
        data = json.load(file)
except Exception:
    data = {"models": []}

d = {}

for k, v in scores.items():
    print(k)
    d[k] = {
        "precision": v["p"],
        "recall": v["r"],
        "f1": v["f"]
    }
    print("\tPrecision:\t", v["p"])
    print("\tRecall:\t\t", v["r"])
    print("\tF1:\t\t", v["f"])
   
data["models"][-1]["scores"] = d 

# Write the updated data back to the file
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)