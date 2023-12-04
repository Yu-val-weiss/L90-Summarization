from typing import Collection, Tuple, Dict, Union
import numpy as np
from tqdm import tqdm
import numpy.typing as npt


class Summarizer:
    @staticmethod
    def _load_vectors(fname="models/wiki-news-300d-1M.vec", num_vectors=-1, specials: Collection[str] = [], first_line_is_n_d=True, dim=300) -> Tuple[Dict[str, int], Dict[int,str], npt.NDArray[np.float32]]:
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            if first_line_is_n_d:
                n, d = map(int, fin.readline().split())
                assert d == dim
            word_index: dict[str, int] = {}
            index_word: dict[int, str] = {}

            vectors = []
            
            if (offset := len(specials)) > 0:
                for ind, spec in enumerate(specials):
                    word_index[spec] = ind
                    index_word[ind] = spec
                    vectors.append(np.zeros(dim))
                    # if spec.lower() == '<pad>':
                    #     vectors.append(np.zeros(dim))
                    # else:
                    #     vectors.append(np.random.uniform(-0.2,0.2,size=dim))

            with tqdm(total=999994 if num_vectors == -1 else num_vectors, desc="Reading embedding vectors") as pbar:
                for index, line in enumerate(fin):
                    tokens = line.rstrip().split(' ')
                    word = tokens[0]
                    vector = np.array(tokens[1:], dtype=float)  # Convert the list of values to a NumPy array
                    word_index[word] = index + offset
                    index_word[index + offset] = word
                    vectors.append(vector)
                    pbar.update(1)
                    if num_vectors > -1 and pbar.n == num_vectors:
                        break

        return word_index, index_word, np.array(vectors)