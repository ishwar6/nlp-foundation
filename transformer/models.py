import torch
# nn is base class for all neural network modules in PyTorch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    '''
    My Custom PyTorch module for input embeddings.
    '''

    def __init__(self, d_model:int, vocub_size:int):
        '''
        d_model: he dimensionality of the embedding space.
        This is the size of each embedding vector. It represents the size of the vector that each token will be mapped to.
        Embeddings transform discrete tokens (like words, characters, or subwords) into continuous vectors. 
        The d_model dimensionality defines how rich or expressive these vectors can be. A higher dimensionality allows the model to potentially capture more information about each token, 
        at the cost of requiring more parameters and computational resources.
        '''
        super().__init__()
        self.d_model = d_model
        self.vocub_size = vocub_size
        self.embidding = nn.Embedding(vocub_size, d_model)


    def forward(self, x):
        '''
        Transforming token indices into dense vectors of a specified size (d_model),
        with the embeddings scaled by the square root of the dimensionality of the embedding space.
        '''
        return self.embidding(x) * math.sqrt(self.d_model)


    
