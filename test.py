from models.extractive_summarizer import ExtractiveSummarizer
import numpy as np


model = ExtractiveSummarizer()

vector1 = model._embed_sentence("Hello I am a big boy".split())

vector2 = model._embed_sentence("Hello I am a very big boy".split())

dot_product = np.dot(vector1, vector2)

# Calculate the magnitudes of each vector
magnitude1 = np.linalg.norm(vector1)
magnitude2 = np.linalg.norm(vector2)

# Calculate the cosine similarity
cosine_similarity = dot_product / (magnitude1 * magnitude2)

print(vector1.shape())
print(vector2.shape())
print(cosine_similarity)