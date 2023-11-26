import random
from typing import List
from evaluation.rouge_evaluator import RougeEvaluator
from tqdm import tqdm
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import os
from models.transformer import Transformer
from models.summarizer import Summarizer
import nltk
from nltk.tokenize import TreebankWordTokenizer
from tabulate import tabulate
    

class AbstractiveSummarizer(Summarizer):

    # model = None

    def __init__(self, learning_rate=0.001, batch_size=32, grad_acc=1, num_epochs=10, keep_best_n_models=2,
                 num_vectors=-1):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.grad_acc = grad_acc
        self.epochs = num_epochs
        self.keep_best_n = keep_best_n_models
        
        nltk.download('punkt')
        self.tokeniser = TreebankWordTokenizer().tokenize
        
        self.word_index, self.emb_vectors = self._load_vectors(num_vectors=num_vectors, specials=["<unk>","<pad>", "<sum>", "</sum>"], index_to_word=True) # index_to_word means dictionary also stores
                                                                                                                                                           # word, index pairs
       
        self.model = Transformer(
            len(self.emb_vectors),
            len(self.emb_vectors),
            d_model=300,
            d_ff=1200,
            heads=6,
            input_embeddings=self.emb_vectors,
            freeze_in=False,
            output_embeddings=self.emb_vectors,
            freeze_out=False,
            pos_enc_max_len=10000
        ) 
        
        self.evaluator = RougeEvaluator()
        
    def train(self, X_raw: List[str], y_raw: List[str], val_X, val_y):
        """
        X: list of strings, each is an article
        y: list of strings, each is a summary
        val_X: list of validation sentences
        learning_rate: learning rate for Adam optimizer
        batch_size: batch size for training
        grad_acc: number of gradient accumulation steps to sum over. Set to > 1 to simulate larger batch sizes on memory-limited machines.
        num_epochs: number of epochs to train for
        keep_best_n_models: number of best models to keep (based on validation performance)
        """

        assert len(X_raw) == len(y_raw), "X and y must have the same length"

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X, y = self.preprocess(X_raw, y_raw)

        best_model_paths = []
        best_model_scores = []

        for epoch in range(self.epochs):
            # Shuffle the dataset:
            num_samples = X.size(0)

            # Generate a list of indices and shuffle it
            indices = list(range(num_samples))
            random.shuffle(indices)
            
            # Shuffle both by same indices
            X, y = X[indices], y[indices]

           # Calculate the total number of batches
            num_batches = num_samples // self.batch_size

            X_batched = torch.chunk(X[:num_batches * self.batch_size], num_batches) # don't accept smaller batch at the end
            y_batched = torch.chunk(y[:num_batches * self.batch_size], num_batches)

            # Train on each batch:
            for idx in tqdm(range(num_batches), desc="Training epoch {}".format(epoch + 1)):
                # Compute the loss:
                loss = self.compute_loss(X_batched[idx], y_batched[idx]) / self.grad_acc

                # Backprop:
                loss.backward()

                # Handle gradient accumulation:
                if (idx % self.grad_acc) == 0:
                    optimizer.step()
                    optimizer.zero_grad()


            # Evaluate the model:
            score = self.compute_validation_score(val_X, val_y)
            
            # print(best_model_scores)

            # Save the model, if performance has improved (keeping n models saved)
            if len(best_model_scores) < self.keep_best_n or score > min(best_model_scores):
                # Save the model:
                best_model_scores.append(score)
                best_model_paths.append("model-" + str(epoch) + "_score-" + str(score) + ".pt")
                torch.save(self.model.state_dict(), best_model_paths[-1])

                # Delete the worst model:
                if len(best_model_scores) > self.keep_best_n:
                    worst_model_index = np.argmin(best_model_scores)
                    
                    os.remove(best_model_paths[worst_model_index])
                    del best_model_paths[worst_model_index]
                    del best_model_scores[worst_model_index]

        # Recall the best model:
        best_model_index = np.argmax(best_model_scores)
        self.model.load_state_dict(torch.load(best_model_paths[best_model_index]))


    def preprocess(self, X, y):
        """
        X: list of strings (i.e., articles)
        y: list of strings (i.e., summaries)
        """
        tok_x = map(lambda x: self.tokeniser(x), X)
        tok_y = map(lambda x: self.tokeniser(x), y)
        
        numericalized_x = [torch.tensor([self.word_index.get(word, self.word_index['<unk>']) for word in sentence]) for sentence in tqdm(tok_x, desc="Preprocessing x")]

        # Add start and end tokens to numericalized_y
        numericalized_y = [torch.tensor([self.word_index['<sum>']] + [self.word_index.get(word, self.word_index['<unk>']) for word in sentence] + [self.word_index['</sum>']]) for sentence in tqdm(tok_y, desc="Preprocessing y")]

        return pad_sequence(numericalized_x, batch_first=True, padding_value=self.word_index['<pad>']), pad_sequence(numericalized_y, batch_first=True, padding_value=self.word_index['<pad>'])
        

    def compute_loss(self, X_batch: Tensor, Y_batch: Tensor):
        """
        X_batch and y_batch have dimensions (batch_size, seq_length, d_model), each has diff seq_length
        """
        
        # print("X batch size:", X_batch.shape)
        # print("y batch size:", Y_batch.shape)
        

        criterion = nn.CrossEntropyLoss()
        
        pad_idx = self.word_index['<pad>']
        # print("Pad idx:", pad_idx)
         # Source Mask
        src_mask = (X_batch != pad_idx).unsqueeze(1)
        
        # print("Loss src mask:", src_mask.shape)

        # Target Mask
        tgt_mask = (Y_batch != pad_idx).unsqueeze(1)
        tgt_mask = tgt_mask & self.subsequent_mask(Y_batch.size(-1))
        
    
        predictions = self.model.forward(X_batch, Y_batch, src_mask, tgt_mask)
        
        # print("Predictions shape: ", predictions.shape)
        gen = self.model.generator(predictions)
        # print("Generator shape:", gen.shape)
        
        # print(gen[0])
        
        # convert Y_batch to a 1-hot vector for each word 
        y_one_hot = F.one_hot(Y_batch, num_classes=len(self.emb_vectors)).to(torch.float)
        
        # print("Y one hot shape:", y_one_hot.shape)
        
        return criterion(gen, y_one_hot)
        
       
    @staticmethod
    def subsequent_mask(size):
        """
        Mask out subsequent positions, i.e. decoder can only use what has already been predicted.
        """
        # Direct quote from annotated implementation.
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0


    def generate(self, X):
        """
        X: list of sentences (i.e., articles)
        """

        self.model.eval()
        
        generated_summaries = []

        for article in tqdm(X, desc="Running abstractive summarizer"):
            with torch.no_grad():
                # Tokenize and numericalize the input article
                input_tokens = self.tokeniser(article)
                input_indices = torch.tensor([self.word_index.get(word, self.word_index['<unk>']) for word in input_tokens]).unsqueeze(0)

                # Source Mask
                src_mask = (input_indices != self.word_index['<pad>']).unsqueeze(1).unsqueeze(2)

                # Generate summary
                output_indices = self.model.predict(input_indices, src_mask, self.word_index['<sum>'], self.word_index['</sum>'])

                # Decode the generated summary
                output_tokens = self.decode(output_indices)
                generated_summaries.append(' '.join(output_tokens))

        return generated_summaries


    def decode(self, tokens: List[int]) -> List[str]:
        """
        tokens: list of token indices
        """
        assert tokens[0] == '<sum>' and tokens[-1] == '</sum>'
        return [self.word_index[token] for token in tokens]

    def compute_validation_score(self, X, y):
        """
        X: list of sentences (i.e., articles)
        y: list of sentences (i.e., summaries)
        """
        # rouge, alternative just use loss
        self.model.eval()
        with torch.no_grad():
            X_p, y_p = self.preprocess(X,y)
            src_mask = (X_p != (pad := self.word_index['<pad>'])).unsqueeze(-2)
            tgt_mask = (y_p != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & self.subsequent_mask(y_p.size(-1)).type_as(
                tgt_mask.data
            )
            out = self.model.forward(X_p, y_p, src_mask, tgt_mask)
            
            x = self.model.generator(out)
            x = torch.argmax(x, dim=-1)
            specials = {"<unk>","<pad>", "<sum>", "</sum>"}
            predicted_words = [[self.word_index[ind] for ind in sent] for sent in x.tolist()]
            # print(predicted_words[0])
            predicted_words = [' '.join([word for word in sent[sent.index("<sum>")+1:sent.index("</sum>") if '</sum>' in sent else len(sent)] if word not in specials]) for sent in predicted_words]
            r = self.evaluator.batch_score(predicted_words, y)
            
            # Convert the data to a list of tuples for tabulate
            table_data = [(key, value['r'], value['p'], value['f']) for key, value in r.items()] # type: ignore

            # Define the headers
            headers = ['Metric', 'Recall', 'Precision', 'F1']

            # Print the table
            table = tabulate(table_data, headers=headers, tablefmt='pretty')
            
            print(table)
            
            rouge_1_f1 = r.get('rouge-1').get('f') # type: ignore
            return rouge_1_f1
            

    def predict(self, X, k=40):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        Y = []
        
        with torch.no_grad():
            
            for article in tqdm(map(lambda x: self.tokeniser(x), X), desc="Running abstractive summarizer", disable=True):
                # print(article)
                # print(len(article))
                article = torch.tensor([self.word_index.get(word, self.word_index['<unk>']) for word in article]).unsqueeze(0)
                print("Article shape:", article.size())
                src_mask = (article != self.word_index['<pad>'])
                print("Src mask shape", src_mask.shape)
                # print(src_mask)

                memory = self.model.encode(article, src_mask)
                print("Memory shape", memory.shape)
                ys = torch.zeros(1, 1).fill_(self.word_index['<sum>']).type_as(article.data)
                print("ys shape: ", ys.shape)
                for step in tqdm(range(k - 1), desc="Predicting article", leave=False, disable=True):
                    out = self.model.decode(
                        memory, src_mask, ys, self.subsequent_mask(ys.size(1)).type_as(article.data)
                    )
                    print("Out shape:", out.shape)
                    prob = self.model.generator(out[:, -1])
                    print("Prob shape:", prob.shape)
                    _, next_word = torch.max(prob, dim=1)
                    print("Next Word:", next_word)
                    next_word = next_word.item()
                    ys = torch.cat(
                        [ys, torch.zeros(1, 1).type_as(article.data).fill_(next_word)], dim=1
                    )
                    print(f"Step {step + 1}: Next Word - {self.word_index[next_word]}, Current Sequence - {ys}")
                    if next_word == self.word_index['</sum>']:
                        break
                Y.append(" ".join(map(lambda x: self.word_index[x],ys[0].tolist()))) # convert from index to string
        return Y