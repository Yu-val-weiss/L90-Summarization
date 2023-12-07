import itertools
import random
from typing import List

import torchtext

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

import fasttext


class AbstractiveSummarizer(Summarizer):

    # model = None

    def __init__(self, learning_rate=0.001, batch_size=32, grad_acc=1, num_epochs=10, keep_best_n_models=2,
                 vocab_size=-1, use_device=True, build_vocab=False, X=None, y=None):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.grad_acc = grad_acc
        self.epochs = num_epochs
        self.keep_best_n = keep_best_n_models

        nltk.download('punkt')
        self.tokeniser = TreebankWordTokenizer().tokenize

        self.specials = ["<unk>","<pad>", "<s>", "<e>"]
        
        self.word_index: dict[str,int] = {}
        self.num_classes = -1
        
        if build_vocab:
            assert X is not None and y is not None
            self.use_lower_case=False
            vocab = self.build_vocab(X, y, max_size=vocab_size if vocab_size > 0 else None)
            self.index_word = {k: v for k, v in enumerate(vocab.get_itos())}
            self.word_index = vocab.get_stoi()
            
            self.model = Transformer(
                len(vocab),
                len(vocab),
                N=4,
                d_model=64,
                d_ff=64*4,
                heads=4,
                pos_enc_max_len=8192,
                same_embeddings=True
            )
            
            self.num_classes = len(vocab)
            
            
            
        else:
            # self.word_index, self.index_word, self.emb_vectors = self._load_vectors(fname='models/glove.6B.100d.txt', first_line_is_n_d=False, dim=100,
            #                                                     num_vectors=vocab_size, specials=self.specials) 
                                                                                                 
            # self.model = Transformer(
            #     len(self.emb_vectors),
            #     len(self.emb_vectors),
            #     N=4,
            #     d_model=self.emb_vectors.shape[-1],
            #     d_ff=self.emb_vectors.shape[-1]*4,
            #     heads=4,
            #     input_embeddings=self.emb_vectors,
            #     freeze_in=False,
            #     same_embeddings=True,
            #     pos_enc_max_len=10000
            # )
            # self.num_classes = len(self.emb_vectors)
            assert X is not None and y is not None
            self.use_lower_case=False
            vocab = self.build_vocab(X, y, max_size=vocab_size if vocab_size > 0 else None, use_lower_case=self.use_lower_case)
            self.index_word = {k: v for k, v in enumerate(vocab.get_itos())}
            self.word_index = vocab.get_stoi()
            
            print("Loading fasttext model...")
            
            ft = fasttext.load_model('models/fasttext.128.bin')
            
            print("...model loaded")
            
            def _embed_word(index):
                if self.index_word[index] in self.specials:
                    return np.zeros(ft.get_dimension())
                return ft.get_word_vector(self.index_word[index])
            
            print("Embedding words...")
            embeddings = np.array([_embed_word(i) for i in range(len(vocab))])
            print("... words embedded, shape: ", embeddings.shape)
            
            self.model = Transformer(
                len(vocab),
                len(vocab),
                N=4,
                d_model=embeddings.shape[-1],
                d_ff=embeddings.shape[-1]*4,
                heads=4,
                input_embeddings=embeddings,
                freeze_in=True,
                same_embeddings=True,
                pos_enc_max_len=10000
            )
            self.num_classes = len(vocab)
            

        self.mps = False
        self.cuda = False

        self.device = torch.device("cpu")

        if use_device:
            if torch.backends.mps.is_available():
                self.mps = True
                self.device = torch.device("mps")
                self.model.float()
                self.model.to(self.device)

            elif torch.cuda.is_available():
                self.mps = False
                self.cuda = True
                self.device = torch.device("cuda")
                self.model.to(self.device)


        self.evaluator = RougeEvaluator()

    def train(self, X_raw: List[str], y_raw: List[str], val_X, val_y, delete_models=False, load_model: str | None = None):
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

        if load_model is not None:
            print(f"Loading model {load_model}...")
            self.model.load_state_dict(torch.load(load_model))
            return

        if delete_models:
            dir = os.listdir()
            for item in dir:
                if item.endswith(".pt"):
                    os.remove(item)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        X, y = self.preprocess(X_raw, y_raw, use_lower_case=self.use_lower_case)

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
                loss = self.compute_loss(X_batched[idx].to(self.device), y_batched[idx].to(self.device)) / self.grad_acc

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
                best_model_paths.append(f"model-{epoch+1}_score-{score:.3f}.pt")
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


    def preprocess(self, *args: List[str], use_lower_case=False):
        """
        args: iterable of List of strings
        """
        def _pr(X: List[str], ind=0):
            # print("Input to preprocessor:", X)
            tok = map(lambda x: self.tokeniser(x), X)
            # print("Tokenized", list(copy.deepcopy(tok)))

            numericalized = [torch.tensor([self.word_index['<s>']] + [self.word_index.get(word.lower() if use_lower_case else word, self.word_index['<unk>'] ) for word in sentence] + [self.word_index['<e>']])
                             for sentence in tqdm(tok, desc=f"Preprocessing arg {ind}")]
            # print("Numericalized size", len(numericalized))
            p_s = pad_sequence(numericalized, batch_first=True, padding_value=self.word_index['<pad>'])
            # print("Padded size", p_s.shape)
            return p_s

        return tuple(_pr(X, ind) for ind, X in enumerate(args))


    def build_vocab(self, *args: List[str], use_lower_case=False, max_size=None):
        def _getwords(X: List[str]):
            tok = map(lambda x: self.tokeniser(x), X)
            return (word.lower() if use_lower_case else word for sentence in tqdm(tok, desc="tokenizing article", leave=False) for word in sentence)
        vocab = torchtext.vocab.build_vocab_from_iterator(itertools.chain(_getwords(X) for X in tqdm(args, desc="Building vocab")), specials=self.specials, min_freq=2, max_tokens=max_size)
        print("Vocab length:", len(vocab))
        return vocab
        
        

    def compute_loss(self, X_batch: Tensor, Y_batch: Tensor):
        """
        X_batch and y_batch have dimensions (batch_size, seq_length, d_model), each has diff seq_length
        """

        # print("X batch size:", X_batch.shape)
        # print("y batch size:", Y_batch.shape)


        criterion = nn.CrossEntropyLoss(label_smoothing=0.3)

        pad_idx = self.word_index['<pad>']

        # print("X device", X_batch.device)
        # print("Y device", Y_batch.device)
        # print("Pad idx:", pad_idx)
         # Source Mask
        src_mask = (X_batch != pad_idx).unsqueeze(1)

        # print("Src mask device", src_mask.device)

        # print("Loss src mask:", src_mask.shape)
        tgt_seq = Y_batch[:, :-1]

        # print("tgt seq device", tgt_seq.device)

        # Target Mask
        tgt_mask = (tgt_seq != pad_idx).unsqueeze(1)
        tgt_mask = tgt_mask & self.subsequent_mask(tgt_seq.size(-1), device=self.device)

        predictions = self.model.forward(X_batch, tgt_seq, src_mask, tgt_mask)

        # print(predictions)

        # print("Predictions shape: ", predictions.shape)
        gen = self.model.generator(predictions)
        # print("Generator shape:", gen.shape)

        # print(gen[0])

        # convert Y_batch to a 1-hot vector for each word
        y_one_hot = F.one_hot(Y_batch[:, 1:], num_classes=self.num_classes).to(torch.float)

        # print("Y one hot shape:", y_one_hot.shape)

        return criterion(gen, y_one_hot)


    @staticmethod
    def subsequent_mask(size, device=None):
        """
        Mask out subsequent positions, i.e. decoder can only use what has already been predicted.
        """
        # Direct quote from annotated implementation.
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0


    def decode(self, tokens: List[int]) -> List[str]:
        """
        tokens: list of token indices
        """
        # assert (start := self.word_index['<s>']) in tokens and (end := self.word_index['<e>']) in tokens
        # return [self.word_index[word] for word in tokens[tokens.index(start)+1:tokens.index(end)] if word not in self.specials]
        return [self.index_word[word] for word in tokens if word not in self.specials]
    

    def compute_validation_score(self, X, y):
        """
        X: list of sentences (i.e., articles)
        y: list of sentences (i.e., summaries)
        """
        # rouge, alternative just use loss
        self.model.eval()
        with torch.no_grad():
            X_p, y_p = self.preprocess(X,y, use_lower_case=self.use_lower_case)
            X_p, y_p = X_p.to(self.device), y_p.to(self.device)
            src_mask = (X_p != (pad := self.word_index['<pad>'])).unsqueeze(-2)
            tgt_mask = (y_p != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & self.subsequent_mask(y_p.size(-1)).type_as(
                tgt_mask.data
            )
            out = self.model.forward(X_p, y_p, src_mask, tgt_mask)

            x = self.model.generator(out)
            x = torch.argmax(x, dim=-1)
            predicted_words = [[self.index_word[ind] for ind in sent] for sent in x.tolist()]
            predicted_words = [' '.join([word for word in sent[sent.index("<s>")+1 if "<s>" in sent else 0:sent.index("<e>") if '<e>' in sent else len(sent)] if word not in self.specials]) for sent in predicted_words]
            # for i in range(len(predicted_words)):
            #     print(f"Prediction {i}:\n", predicted_words[i])
            #     print(f"Truth {i}:\n", y[i])
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
        ys = torch.zeros(1, 1, device=self.device).fill_(start_symbol).type_as(src.data)
        for i in (pbar:=tqdm(range(max_len - 1), desc="Generating greedy symbols", leave=False)):
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


    def beam_search_decode(self, src, src_mask, max_len, start_symbol, beam_size):
        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1, device=self.device).fill_(start_symbol).type_as(src.data)

        # Initialize the beam with the start symbol
        beams = [(ys, 0)]  # (current_sequence, log_prob)

        for step in tqdm(range(max_len - 1), desc="Generating beam symbols", leave=False):
            new_beams = []

            for current_seq, log_prob in beams:
                # Extend the current sequence
                out = self.model.decode(
                    memory,
                    src_mask,
                    current_seq,
                    self.subsequent_mask(current_seq.size(1)).type_as(src.data),
                )
                prob = self.model.generator(out[:, -1])
                log_probs, next_words = torch.topk(prob, beam_size, dim=1)

                for i in range(beam_size):
                    new_seq = torch.cat([current_seq, next_words[:, i:i + 1]], dim=1)
                    new_log_prob = log_prob + log_probs[:, i]

                    new_beams.append((new_seq, new_log_prob.item()))

            # Select the top-k sequences
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = new_beams

        # Return the best sequence
        best_sequence, _ = max(beams, key=lambda x: x[1])
        return best_sequence


    def nucleus_decode(self, src, src_mask, max_len, start_symbol, top_p: float):
        memory = self.model.encode(src, src_mask)
        ys = torch.zeros(1, 1, device=self.device).fill_(start_symbol).type_as(src.data)
        for i in (pbar:=tqdm(range(max_len - 1), desc="Generating nucleus symbols", leave=False)):
            out = self.model.decode(
                memory, src_mask, ys, self.subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = self.model.generator(out[:,-1], True)

            # sample next word from top p distribution
            sorted_probs, sorted_indices = torch.sort(prob, descending=True, dim=-1)
            # print("Sorted indices", sorted_indices.shape)
            # print("Sorted probs", sorted_probs.shape)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # need exp since log_softmax used by generator
            mask = cumulative_probs <= top_p
            truncated_probs = torch.where(mask, sorted_probs, torch.tensor(0.0).to(self.device))
            truncated_probs = F.softmax(truncated_probs, dim=-1)

            sampled_index = torch.multinomial(truncated_probs, 1).item()

            # print(sampled_index)

            next_word = sorted_indices[:, int(sampled_index)]

            next_word = next_word.item()

            # print(f"{next_word}: {self.index_word[int(next_word)]}")
            # pbar.write("Next word index: " + str(next_word))
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys
    
    
    def top_k_decode(self, src, src_mask, max_len, start_symbol, top_k: int):
        memory = self.model.encode(src, src_mask)
        ys = torch.zeros(1, 1, device=self.device).fill_(start_symbol).type_as(src.data)
        for i in (pbar:=tqdm(range(max_len - 1), desc="Generating top-k symbols", leave=False)):
            out = self.model.decode(
                memory, src_mask, ys, self.subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = self.model.generator(out[:,-1], True)

            # sample next word from top-k distribution
            sorted_probs, sorted_indices = torch.topk(prob, top_k, dim=-1)
            # print("Sorted indices", sorted_indices.shape)
            # print("Sorted probs", sorted_probs.shape)

            # Normalize probabilities
            truncated_probs = F.softmax(sorted_probs, dim=-1)

            # Sample from the top-k candidates
            sampled_index = torch.multinomial(truncated_probs, 1).item()

            next_word = sorted_indices[:, int(sampled_index)]

            next_word = next_word.item()

            # print(f"{next_word}: {self.index_word[int(next_word)]}")
            # pbar.write("Next word index: " + str(next_word))
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys


    def predict(self, X, k=40, nucleus_decode=False, greedy_decode=False, beam_search=False, top_k=False):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        Y = []

        (X,) = self.preprocess(X)
        X = X.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for src in tqdm(X, desc="Running abstractive summarizer", disable=False):
                # print("Src shape", src.shape)
                # print("Src", src)
                src.unsqueeze_(0)
                max_len = src.size(-1) // 10
                src_mask = torch.ones(1,1,src.size(-1)).to(self.device)
                y_1 = self.nucleus_decode(src, src_mask, max_len, 2, 0.5) if nucleus_decode else torch.tensor([])
                # print(y_1)
                y_2 = self.greedy_decode(src, src_mask, max_len, 2) if greedy_decode else torch.tensor([])
                # print(y_2)
                y_3 = self.beam_search_decode(src, src_mask, max_len, 2, 25) if beam_search else torch.tensor([])
                
                y_4 = self.top_k_decode(src, src_mask, max_len, 2, 40) if top_k else torch.tensor([])
                # print(y_3)
                # print(y)
                y = {
                    "top_p": self.decode(y_1.squeeze().tolist()), 
                    "greedy": self.decode(y_2.squeeze().tolist()),
                    "beam": self.decode(y_3.squeeze().tolist()),
                    "top_k": self.decode(y_4.squeeze().tolist()),
                }
                print(y)
                Y.append(y)

        return Y