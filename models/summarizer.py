from typing import Collection, Tuple, Dict
import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from collections import defaultdict


class Summarizer:
    @staticmethod
    def _load_vectors(fname="models/wiki-news-300d-1M.vec", num_vectors=-1, specials: Collection[str] = []) -> Tuple[Dict[str, int], npt.NDArray[np.float32]]:
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            n, d = map(int, fin.readline().split())
            word_index = {}

            vectors = []
            
            if (offset := len(specials)) > 0:
                for ind, spec in enumerate(specials):
                    word_index[spec] = ind
                    if spec.lower() == '<pad>':
                        vectors.append(np.zeros(d))
                    else:
                        vectors.append(np.random.uniform(-0.3,0.3,size=d))

            with tqdm(total=999994 if num_vectors == -1 else num_vectors, desc="Reading embedding vectors") as pbar:
                for index, line in enumerate(fin):
                    tokens = line.rstrip().split(' ')
                    word = tokens[0]
                    vector = np.array(tokens[1:], dtype=float)  # Convert the list of values to a NumPy array
                    word_index[word] = index + offset
                    vectors.append(vector)
                    pbar.update(1)
                    if num_vectors > -1 and pbar.n == num_vectors:
                        break

        return word_index, np.array(vectors)