# Summarization Exercise

This repository contains code to be used in the L90 assignments on summarization.

## Dataset

We provide a subset of the CNN/Daily Mail dataset to use for training and evaluation. The data is represented as JSON, with the following format:

```
{
    "article" : "The article to be summarized.",
    "summary" : "The desired summary.",
    "greedy_n_best_indices" : "Binary yes/no decisions for each sentence representing whether they should be included in the greedily chosen best extractive summary. Sentences split on periods. Only included for train.greedy_sent.json."
}
```

## Scripts

We provide several scripts to help you get started. To train and predict from an extractive summarizer, run the following (you may want to implement the code in models/extractive_summarizer.py first):

```
python run_extractive_summarizer.py --eval_data dataset_to_predict_for.json > prediction_file.json
```

To evaluate your predictions, run the following:

```
python eval.py --eval_data dataset_to_predict_for.json --pred_data prediction_file.json
```

For assignment 3, we also provide some scripts to help you get started building an abstractive summarizer. Once you have implemented the code in models/abstractive summarizer, you can run it as follows:

```
python run_abstractive_summarizer.py --eval_data dataset_to_predict_for.json > prediction_file.json
```

Please note that you are welcome to ignore our scripts - they are just a scaffolding you can build from if you find it helpful.

## Installation

To run our scripts, please run the following commands to install libraries for evaluation and pretty progress bars:

```
pip install tqdm
pip install rouge_metric
```

For assignment 3, please also install:

```
pip install numpy
```

Please also install PyTorch, following the specific instructions for your machine at https://pytorch.org/get-started/locally/.