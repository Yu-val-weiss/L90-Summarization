import copy
import random
from typing import List, Tuple

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
        
        self.specials = ["<unk>","<pad>", "<s>", "<e>"]
        
        self.word_index, self.emb_vectors = self._load_vectors(num_vectors=num_vectors, specials=self.specials, index_to_word=True) # index_to_word means dictionary also stores
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


    def preprocess(self, *args: List[str]):
        """
        args: iterable of List of strings
        """
        def _pr(X: List[str], ind=0):
            # print("Input to preprocessor:", X)
            tok = map(lambda x: self.tokeniser(x), X)
            # print("Tokenized", list(copy.deepcopy(tok)))
            numericalized = [torch.tensor([self.word_index['<s>']] + [self.word_index.get(word, self.word_index['<unk>'] ) for word in sentence] + [self.word_index['<e>']]) for sentence in tqdm(tok, desc=f"Preprocessing arg {ind}")]
            # print("Numericalized size", len(numericalized))
            p_s = pad_sequence(numericalized, batch_first=True, padding_value=self.word_index['<pad>'])
            # print("Padded size", p_s.shape)
            return p_s
            
        return tuple(_pr(X, ind) for ind, X in enumerate(args))
        

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
        tgt_seq = Y_batch[:, :-1]
        # Target Mask
        tgt_mask = (tgt_seq != pad_idx).unsqueeze(1)
        tgt_mask = tgt_mask & self.subsequent_mask(tgt_seq.size(-1))
    
        predictions = self.model.forward(X_batch, tgt_seq, src_mask, tgt_mask)
        
        # print(predictions)
        
        # print("Predictions shape: ", predictions.shape)
        gen = self.model.generator(predictions)
        # print("Generator shape:", gen.shape)
        
        # print(gen[0])
        
        # convert Y_batch to a 1-hot vector for each word 
        y_one_hot = F.one_hot(Y_batch[:, 1:], num_classes=len(self.emb_vectors)).to(torch.float)
        
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


    def decode(self, tokens: List[int]) -> List[str]:
        """
        tokens: list of token indices
        """
        assert (start := self.word_index['<s>']) in tokens and (end := self.word_index['<e>']) in tokens
        return [self.word_index[word] for word in tokens[tokens.index(start)+1:tokens.index(end)] if word not in self.specials]

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
            predicted_words = [[self.word_index[ind] for ind in sent] for sent in x.tolist()]
            predicted_words = [' '.join([word for word in sent[sent.index("<s>")+1 if "<s>" in sent else 0:sent.index("<e>") if '<e>' in sent else len(sent)] if word not in self.specials]) for sent in predicted_words]
            # print(predicted_words)
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
        
        
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        memory = self.model.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for i in (pbar:=tqdm(range(max_len - 1), desc="Generating symbols", leave=False)):
            out = self.model.decode(
                memory, src_mask, ys, self.subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = self.model.generator(out[:,-1])
            # print(prob)
            _, next_word = torch.max(prob, dim=1)
            # print(next_word)
            next_word = next_word.data[0]
            # pbar.write("Next word index: " + str(next_word))
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys
            

    def predict(self, X, k=40):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        Y = []
        
        (X,) = self.preprocess(X)
        
        self.model.eval()
        with torch.no_grad():
            for src in tqdm(X, desc="Running abstractive summarizer", disable=True):
                print("Src shape", src.shape)
                print("Src", src)
                src.unsqueeze_(0)
                max_len = src.size(-1) // 10
                src_mask = torch.ones(1,1,src.size(-1))
                y = self.greedy_decode(src, src_mask, max_len, 2)
                print(y)
                Y.append(self.decode(y.squeeze().tolist()))
                
        return Y