from random import shuffle
from typing import Dict, List, Set, Tuple
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
import json
from datetime import datetime
from evaluation.rouge_evaluator import RougeEvaluator

LEAD_TRAIL_PUNC_REGEX = r"^[^\w\s]+|[^\w\s]+$"

class ExtractiveSummarizer:
    
    def __init__(self, skip_vectors=False, force_idf=False, less_vectors=False) -> None:
        if not skip_vectors:
            self.word_index, self.vectors = self._load_vectors(less_vectors=less_vectors)
        self.inv_doc_frq: Dict[str, float] = {}
        self.force_idf = force_idf
    
    @staticmethod
    def _load_vectors(fname="models/wiki-news-300d-1M.vec", less_vectors=False) -> Tuple[Dict[str, int], npt.NDArray[np.float32]]:
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            n, d = map(int, fin.readline().split())
            word_index = {}

            vectors = []
            
            SHORT = 1000

            with tqdm(total=999994 if not less_vectors else SHORT, desc="Reading embedding vectors") as pbar:
                for index, line in enumerate(fin):
                    tokens = line.rstrip().split(' ')
                    word = tokens[0]
                    vector = np.array(tokens[1:], dtype=float)  # Convert the list of values to a NumPy array
                    word_index[word] = index
                    vectors.append(vector)
                    pbar.update(1)
                    if less_vectors and pbar.n == SHORT:
                        break

        return word_index, np.array(vectors)
        
    
    def _tf(self, article: List[str]) -> npt.NDArray[np.float32]:
        counter: Counter[str] = Counter()
        for sent in article:
            counter.update(map(lambda x: x.lower(), sent))
        _, max_count = counter.most_common(1)[0]
        per_sentence_tf: List[float] = []
        for sent in article:
            if sent == '':
                per_sentence_tf.append(0)
                continue
            per_sentence_tf.append(np.max([(0.5 + 0.5 * (counter[word.lower()] / max_count)) for word in sent]))
        return np.array(per_sentence_tf)
    
    
    def _per_sent_idf(self, article: List[str]) -> npt.NDArray[np.float32]:   
        per_sentence_idf: List[float] = []
        cleaned_article = [[regexed for word in sentence.split() if (regexed := re.sub(LEAD_TRAIL_PUNC_REGEX, '', word.lower())).strip()] for sentence in article]
        for sent in cleaned_article:
            if sent == []:
                per_sentence_idf.append(0)
                continue
            per_sentence_idf.append(np.max(np.array([self.inv_doc_frq[word] if word in self.inv_doc_frq else 0 for word in sent])))
        return np.array(per_sentence_idf)
    
    
    def _make_tf_idf_tfidf(self, article: List[str]):
        cleaned_article = [[re.sub(LEAD_TRAIL_PUNC_REGEX, '', word.lower()).strip() for word in sentence.split()] for sentence in article]
        
        # Compute word counts for the entire article
        word_counts = Counter(word for sentence in cleaned_article for word in sentence)
        max_count = max(word_counts.values())
        
        idf, tf, tfidf = [], [], []
        
        for sent in cleaned_article:
            if sent == []:
                idf.append(0)
                tf.append(0)
                tfidf.append(0)
                continue
            idf.append(sent_tf := np.array([self.inv_doc_frq[word] if word in self.inv_doc_frq else 0 for word in sent]))
            tf.append(sent_idf := np.array([0.5 + 0.5 * (word_counts[word.lower()] / max_count) for word in sent]))
            tfidf.append(sent_tf * sent_idf)
        
        return [np.max(i) for i in idf], [np.max(i) for i in tf], [np.max(i) for i in tfidf]

    
    def _init_idf(self, X: List[List[str]]):
        clean = [[re.sub(LEAD_TRAIL_PUNC_REGEX,'',word.lower()) for sentence in article for word in sentence.split()] for article in X]
        N = len(X)
        for article in tqdm(clean, desc="Generating idf for articles"):
            for word in tqdm(article, desc="Generating idf for words in article", leave=False):
                if word not in self.inv_doc_frq:
                    self.inv_doc_frq[word] = np.log((1+N) / (1 + sum(1 for article in clean if word in article)))
                    
    
    def calculate_idf(self, articles: List[Set[str]], words: List[str], N, id):
        idf_values = {}
        for word in tqdm(words, desc=f"Doing words in group {id}", leave=None): # guaranteed unique due to cleaning
            idf = np.log((1 + N) / (1 + sum(1 for a in articles if word in a)))
            idf_values[word] = idf
        return idf_values

    def _init_idf_parallel(self, X: List[List[str]]):
        if not self.force_idf:
            print("Attempting to load idf json...")
            try:
                with open("models/idf.json", "r", encoding="utf-8") as f:
                    self.inv_doc_frq = json.load(fp=f)    
                print("JSON loaded!")
                if len(self.inv_doc_frq) > 0:
                    print("IDF populated")
                    return
                else:
                    print("IDF empty")
            except Exception as e:
                print(f"JSON not loaded, exception: {e}")
        
        else: 
            print("Re-train forced")
        
        clean = [
            {
                regexed # Remove non-word characters
                for sentence in article
                for word in sentence.split()
                if (regexed := re.sub(LEAD_TRAIL_PUNC_REGEX, '', word.lower())).strip() # Skip empty words, and clean leading/trailing punctuation
            }
            for article in tqdm(X, desc="Cleaning articles")
        ]
        corpus_words = set()
        for article in tqdm(clean, desc="Finding all unique words"):
            corpus_words |= article
        print(f"{(num_words := len(corpus_words))} unique words found")
        
        # split words into groups
        NUM_THREADS = 8
        
        seg_size = num_words // NUM_THREADS
        
        corpus_words = list(corpus_words)
        
        partitions = [corpus_words[i:i + seg_size] for i in range(0, num_words, seg_size)]
        
        N = len(X)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.calculate_idf, clean, words, N, ind) for ind, words in enumerate(partitions)]
            
            # Combine idf values from all threads
            done, _ = wait(futures)
            [self.inv_doc_frq.update(future.result()) for future in done]
                    
        print("Dumping json...")
        with open("models/idf.json", "w", encoding="utf-8") as f:
            json.dump(self.inv_doc_frq, fp=f, ensure_ascii=False, indent=4)
        print("JSON dumped!")
                    
                
    def _embed_sentence(self, sent: str):
        def random_embedding():
            scale = np.sqrt(1 / 300) # dimension size
            return np.random.uniform(-scale, scale, 300)
        STOP_WORDS = ['the', 'a', 'an']
        cleaned_sentence = [re.sub(LEAD_TRAIL_PUNC_REGEX,'',word) for word in sent.split() if word.lower() not in STOP_WORDS]
        if cleaned_sentence == []:
            cleaned_sentence = [re.sub(LEAD_TRAIL_PUNC_REGEX,'',word) for word in sent.split() if word.lower()] # if only made up of stop words, include its
        if cleaned_sentence == []:
            return random_embedding()
        # print(cleaned_sentence)
        return np.mean([(self.vectors[self.word_index[word]] if word in self.word_index else random_embedding()) for word in cleaned_sentence], axis=0)
        
        
    def _embed_article(self, article: List[str]):
        x = [emb for sent in article if (emb := self._embed_sentence(sent)).shape == (300,)]
        if len(x) == 0:
            print(article)
        return np.mean(x, axis=0)
            
        
    
    @staticmethod
    def _cosine_distance(v1: np.ndarray, v2: np.ndarray):
        mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
        return np.dot(v1, v2) / (mag1 * mag2)     

    def preprocess(self, X) -> List[List[str]]:
        """
        X: list of list of sentences (i.e., comprising an article), i.e. list of articles, where each article is a list of sentences
        """
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        return split_articles
    
    @staticmethod
    def log_loss(predictions, y):
        epsilon = 1e-20  # Small constant to prevent log(0)
        return -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    
    @staticmethod
    def sigmoid(x): # numerically stable
        return np.where(x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x)))
    
    @staticmethod
    def sigmoid_prime(z):
        return ExtractiveSummarizer.sigmoid(z) * (1-ExtractiveSummarizer.sigmoid(z))
    
    
    def create_feature_for_article(self, article: List[str]):
        embedded_article = self._embed_article(article)
        cosines = np.array([self._cosine_distance(emb, embedded_article) if (emb := self._embed_sentence(sent)).shape == (300,) else -1.0 for sent in article])
        tf, idf, tfidf = self._make_tf_idf_tfidf(article)
        # position = np.arange(1, len(article) + 1)
        # return np.column_stack((cosines, tf, idf, tfidf, position / np.linalg.norm(position)))
        return np.column_stack((cosines, tf, idf, tfidf))
        # return np.column_stack((cosines, tf, idf))
    
    
    def calculate_derivatives_for_batch(self, weights, bias, feat_batch, y_batch):
        w_deriv: npt.NDArray = np.zeros(3)
        b_deriv = np.float64(0)
        for features, gold_y in zip(feat_batch, y_batch): # iterating per article here, use average loss in the article
            raw =  features.dot(weights) + bias
            norm = self.sigmoid(raw)
            
            # print(f"raw: {raw}")
            # print(f"norm: {norm}")
        
            # cross entropy loss derivatives:s
            # ∂L/∂w = [σ(z) - y] * x
            # ∂L/∂b = [σ(z) - y]
            # print(f"gold_y: {gold_y}")
            
            diff = norm - gold_y
            
            # print(f"diff: {diff}")
            
            max_deriv = np.max(diff)
            
            # print(f"avg_deriv: {avg_deriv}")
            w_diff = max_deriv * np.amax(features, axis=0)
            # print(f"w_diff: {w_diff}")
            w_deriv += w_diff
            b_deriv += max_deriv
            
        return w_deriv, b_deriv
    
    
    def train(self, X: List[List[str]], y: List[List[bool]]):
        """
        X: list of list of sentences (i.e., comprising an article)
        y: list of yes/no decision for each sentence (as boolean)
        """

        for article, decisions in tqdm(zip(X, y), desc="Validating data shape", total=len(X)):
            assert len(article) == len(decisions), "Article and decisions must have the same length"
            
        self._init_idf_parallel(X)
        
        # prepare features: cosine distance of each sentence to article, tf, idf, tfidf, pos in article
        # these remain constant, only weights will change
        
        all_features = [self.create_feature_for_article(article) for article in tqdm(X, desc="Preparing features")]
            
        # now can perform training
        EPOCHS = 1000
        LEARNING_RATE = 0.001
        FEATURE_COUNT = 4
        BATCH_SIZE = 100
        EARLY_STOP = 2
        LAMBDA = 0.8
        
        weights: npt.NDArray[np.float64] = np.random.uniform(0, 1, FEATURE_COUNT)
        bias: np.float64 = np.float64(0.0)
        
        evaluator = RougeEvaluator()
        
        best_p, best_r, best_f = 0.0, 0.0, 0.0
        early_stop_p, early_stop_r, early_stop_f = 0, 0, 0
        
        for epoch in tqdm(range(EPOCHS), desc="Descending gradients"):
            w_deriv: npt.NDArray = np.zeros(FEATURE_COUNT)
            b_deriv = np.float64(0)
            
            # Shuffle the data and labels to create random mini-batches
            combined_data = list(zip(all_features, y))
            shuffle(combined_data)
            all_features, y = zip(*combined_data)
            
            for i in range(0, len(all_features), BATCH_SIZE):
                batch_features = all_features[i:i + BATCH_SIZE]
                batch_labels = y[i:i + BATCH_SIZE]
                
                for features, gold_y in zip(batch_features, batch_labels):
                    raw = features.dot(weights) + bias
                    norm = self.sigmoid(raw)
                    diff = norm - gold_y
                    avg_deriv = np.amax(diff)
                    w_diff = avg_deriv * np.amax(features, axis=0)
                    w_deriv += w_diff
                    b_deriv += avg_deriv

                # update weights and bias for mini-batch
                weights -= (w_deriv - LAMBDA * weights) * LEARNING_RATE / BATCH_SIZE
                bias -= (b_deriv - LAMBDA * bias) * LEARNING_RATE / BATCH_SIZE
                
                
            # check for early stop using very_small_validation, every N epochs
            if EARLY_STOP > 0 and epoch != 0 and epoch % 100 == 0:
                with open('data/very_small_validation.json', 'r') as f:
                    eval_data = json.load(f)
                    

                eval_articles = [article['article'] for article in eval_data]
                preprocessed_eval_articles = self.preprocess(eval_articles)
                summaries = self.predict(preprocessed_eval_articles, b=bias, w=weights)
                    
                pred_sums = []
                eval_sums = []
                for eval, pred in zip(eval_data, summaries):
                    pred_sums.append(pred)
                    eval_sums.append(eval['summary'])
                
                scores = evaluator.batch_score(pred_sums, eval_sums) 
                
                for k, v in scores.items(): # type: ignore
                    if k == 'rouge-1':
                        if v["f"] > best_f:
                            best_f = v["f"]
                            early_stop_f = 0  # Reset the counter
                        else:
                            early_stop_f += 1
                        if v["p"] > best_p:
                            best_p = v["p"]
                            early_stop_p = 0  
                        else:
                            early_stop_p += 1
                        if v["r"] > best_r:
                            best_r = v["r"]
                            early_stop_r = 0  
                        else:
                            early_stop_r += 1
                            
                count = 0
                for item in (early_stop_r, early_stop_p, early_stop_f):
                    if item >= EARLY_STOP:
                        count += 1
                if count >= 2:
                    print("STOPPING EARLY")
                    break 
        
            
        # write learned results
        self.weights = weights
        self.bias = bias
        
        print(f"Weights: {weights}")
        print(f"Bias: {bias}")
        
        file_path = "models/weights.json"
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except Exception:
            data = {"models": []}

        # Concatenate the arrays
        data["models"].append({
            "weights": weights.tolist(),
            "bias": bias,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tag": "short" if len(X) == 100 else "full",
            "learning_rate": LEARNING_RATE,
            "epoch": EPOCHS,
            "feature_count": FEATURE_COUNT,
            "batch_size": BATCH_SIZE,
            "early_stop": EARLY_STOP,
            "lambda": LAMBDA
        })

        # Write the updated data back to the file
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def predict(self, X: List[List[str]], k=3, b=None, w=None):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        # print(self.create_feature_for_article([" ", "afhash asdf"]))
        
        bias = b if b is not None else self.bias
        weights = w if w is not None else self.weights
        
        loop = tqdm(X, desc="Running extractive summarizer") if b is None else X
        
        for article in loop:
            
            features = self.create_feature_for_article(article)
            raw = bias + features.dot(weights)
            sentence_scores = self.sigmoid(raw)
            
            # print(sentence_scores)

            # Pick the top k sentences as summary.
            # Note that this is just one option for choosing sentences.
            top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)
            top_sentences = [article[i] for i in top_k_idxs if article[i].strip() and not re.match(r"^[^\w\s]*$", article[i])][:k]
            summary = ' . '.join(top_sentences)
            
            yield summary