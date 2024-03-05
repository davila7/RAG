import cohere
import numpy as np
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)

# get the embeddings
words = ["Red", "Blood", "Sea"]
(p1, p2, p3) = co.embed(words).embeddings

# compare them
def calculate_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Similarity -1 to 1

#print(calculate_similarity(p1, p2)) 

print(calculate_similarity(p1, p3)) 