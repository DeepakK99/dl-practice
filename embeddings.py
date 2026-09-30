import torch
import torch.nn as nn

# Suppose we have a vocabulary size of 5 tokens.
# We want to map each token to a 4-dimensional dense vector (Embedding size).
vocab_size = 5
embedding_dim = 4

# 1. Initialize the embedding layer (Think of this as a trainable matrix of size 5x4)
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Look at the raw weights inside this layer
print("Initial Token-to-Vector Lookup Table Matrix:")
print(embedding_layer.weight.data)

# 2. Simulate an input batch of tokenized sentences.
# Sentence 1: "king queen" -> [1, 2]
# Sentence 2: "apple banana" -> [3, 4]
mock_input_tokens = torch.tensor([[1, 2], 
                                   [3, 4]], dtype=torch.long)

# 3. Pass token IDs through the embedding layer
dense_vectors = embedding_layer(mock_input_tokens)

print("\nOutput Dense Vector Shape:", dense_vectors.shape)
# Shape: [2 batches, 2 tokens per batch, 4 embedding dimensions per token]

print("\nVector for 'king' (First batch, first token):")
print(dense_vectors[0][0])
