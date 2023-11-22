from typing import List
import tqdm
import random
import torch
import numpy as np
import os
from models.transformer import Transformer
from models.summarizer import Summarizer
import nltk
from nltk.tokenize import TreebankWordTokenizer
import torchtext.vocab
from collections import Counter, OrderedDict

class AbstractiveSummarizer(Summarizer):

    model = None

    def __init__(self, learning_rate=0.001, batch_size=32, grad_acc=1, num_epochs=10, keep_best_n_models=2,
                 num_vectors=-1):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.grad_acc = grad_acc
        self.epochs = num_epochs
        self.keep_best_n = keep_best_n_models
        
        nltk.download('punkt')
        self.tokeniser = TreebankWordTokenizer().tokenize
        
        self.word_index, self.emb_vectors = self._load_vectors(num_vectors=num_vectors, specials=["<unk>","<pad>", "<sum>", "</sum>"])
       
        self.model = Transformer(
            len(self.word_index),
            len(self.word_index),
            d_model=300,
            d_ff=1200,
            heads=6,
            input_embeddings=self.emb_vectors,
            freeze_in=False,
            output_embeddings=self.emb_vectors,
            freeze_out=False,
        ) 
        
    def train(self, X: List[str], y: List[str], val_X, val_y):
        """
        X: list of strings, each is an article (i.e., articles)
        y: list of strings, each is a summary
        val_X: list of validation sentences
        learning_rate: learning rate for Adam optimizer
        batch_size: batch size for training
        grad_acc: number of gradient accumulation steps to sum over. Set to > 1 to simulate larger batch sizes on memory-limited machines.
        num_epochs: number of epochs to train for
        keep_best_n_models: number of best models to keep (based on validation performance)
        """

        assert len(X) == len(y), "X and y must have the same length"

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        dataset = self.preprocess(X, y)

        best_model_paths = []
        best_model_scores = []

        for epoch in range(num_epochs):
            # Shuffle the dataset:
            np.random.shuffle(dataset)

            # Split into batches:
            batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

            # Train on each batch:
            for idx, batch in tqdm.tqdm(enumerate(batches), desc="Training epoch {}".format(epoch)):
                # Compute the loss:
                loss = self.compute_loss(batch) / grad_acc

                # Backprop:
                loss.backward()

                # Handle gradient accumulation:
                if (idx % grad_acc) == 0:
                    optimizer.step()
                    optimizer.zero_grad()


            # Evaluate the model:
            score = self.compute_validation_score(val_X, val_y)

            # Save the model, if performance has improved (keeping n models saved)
            if len(best_model_scores) < keep_best_n_models or score > min(best_model_scores):
                # Save the model:
                best_model_scores.append(score)
                best_model_paths.append("model-" + str(epoch) + "_score-" + str(score) + ".pt")
                torch.save(self.model.state_dict(), best_model_paths[-1])

                # Delete the worst model:
                if len(best_model_scores) > keep_best_n_models:
                    worst_model_index = np.argmin(best_model_scores)
                    
                    os.remove(best_model_paths[worst_model_index])
                    del best_model_paths[worst_model_index]
                    del best_model_scores[worst_model_index]

        # Recall the best model:
        best_model_index = np.argmax(best_model_scores)
        self.model.load_state_dict(torch.load(best_model_paths[best_model_index]))


    def preprocess(self):
        """
        X: list of strings (i.e., articles)
        y: list of strings (i.e., summaries)
        """

        self.

    def compute_loss(self, batch):
        """
        batch: tensor of token indices. Dimensionality is (batch_size, sequence_length)
        """

        # TODO: Implement me!

    def generate(self, X):
        """
        X: list of sentences (i.e., articles)
        """

        self.model.eval()
        
        # TODO: Implement me!


    def decode(self, tokens):
        """
        tokens: list of token indices
        """

        # TODO: Implement me!

    def compute_validation_score(self, X, y):
        """
        X: list of sentences (i.e., articles)
        y: list of sentences (i.e., summaries)
        """

        self.model.eval()

        # TODO: Implement me. You can compute either loss or ROUGE.

    def predict(self, X, k=3):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        for article in tqdm.tqdm(X, desc="Running abstractive summarizer"):
            """
            TODO: Implement me!
            """

            output_token_indices = self.generate(article)
            output_tokens = self.decode(output_token_indices)
            summary = ' . '.join(output_tokens)
            
            yield summary