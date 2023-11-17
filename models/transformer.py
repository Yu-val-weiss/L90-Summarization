from math import log, sqrt
from typing import Type, Union
from torch import Tensor, dropout, nn
import torch

# default pytorch is (seq_length, batch_size, data_size)
# alternative is (batch, seq, data)
# thinking I will use batch first as most intuitive to me

def clonelayer(N: int, layer: Type[nn.Module], *args, **kwargs):
    '''
    Produces N identical instances of the layer class. 
    args and kwargs are arguments to the constructor of layer. 
    '''
    return nn.ModuleList([layer(*args, **kwargs) for _ in range(N)])

class LayerNormaliser(nn.Module):
    '''
    Performs layer normalisation on the input
    '''
    def __init__(self, d_model: int, epsilon=1e-5) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon  # prevent division by zero
        
    def forward(self, X: Tensor):
        sigma, mu = torch.std_mean(X, dim=-1, keepdim=True) 
        return (self.gain * (X - mu) / (sigma + self.epsilon)) + self.bias
    

class AddAndNorm(nn.Module):
    '''
    Defines add and normalisation for a sublayer within for an encoder or decoder layer
    '''
    def __init__(self, d_model, dropout) -> None:
        super().__init__()
        self.layer_norm = LayerNormaliser(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X: Tensor, sublayer: nn.Module):
        '''
        Applies dropout to the output of each sub-layer, before it is added to the sub-layer input and normalised.
        The annotated solution applies normalisation first for 'code simplicity', so I do so as well. 
        '''
        return X + self.dropout(sublayer(self.layer_norm(X)))
    
    
class PositionWiseFeedForward(nn.Module):
    '''
    Each of the layers in the encoder and decoder contains a fully connected feed-forward network.
    '''
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1) -> None:
        super().__init__()
        self.l_1 = nn.Linear(d_model, d_ff)
        self.l_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, X: Tensor):
        return self.l_2(self.dropout(self.relu(self.l_1(X))))
    

class TransformerEmbeddings(nn.Module):
    '''
    Embeddings for the transformer. Can be initialised from a pretrained numpy array. 
    '''
    def __init__(self, embedding_dim, vocab_size, from_pretrained=None, freeze=True):
        self.embedding_dim = embedding_dim
        if from_pretrained is not None:
            assert from_pretrained.shape == (vocab_size, embedding_dim)
            self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(from_pretrained), freeze=freeze)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, X: Tensor):
        return self.embeddings(X) * torch.sqrt(self.embedding_dim)
    

class PositionalEncoding(nn.Module):
    '''
    A way of encoding the position of the token in the same dimensions as the word embeddings.
    '''
    
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pos_enc = torch.zeros(max_len, embedding_dim)
        
        position = torch.arange(max_len).unsqueeze(1) # unsqueeze turns 1 row of length max_len into max_len rows of length 1
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-log(10000.0) / embedding_dim) # negative exponent so don't divide but multiply
        ) # use log rule to convert exponent to prevent rounding issues (makes zeros otherwise)
        
        pos_enc[:, 0::2] = torch.sin(position * div_term) # the slice means for all rows, all even index columns
        pos_enc[:, 1::2] = torch.cos(position * div_term) # " "                               odd  
        pos_enc = pos_enc.unsqueeze(0) # if using batch first, use (0) else use (1)
        self.register_buffer("pos_enc", pos_enc) # this stores the pos_enc, but not as a parameter to be trained
        
    def forward(self, X: Tensor):
        # trim positional encodings to the actual length of the sequence
        X = X + self.pe[:, :X.size(1)].requires_grad_(False) # this assumes batch first, otherwise use self.pe[:X.size(0)]
        return self.dropout(X)
    
    
class MultiAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout=0.1):
        super().__init__()
        # d_k = d_v = d_model/h
        assert d_model % heads == 0
        # for the sake of this, d_k and d_v are always the same
        self.d_k = d_model // heads
        self.heads = heads
        self.linears = clonelayer(4, nn.Linear, d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Union[Tensor, None]=None):
        # quoted from annotated example
        if mask is not None:
            # to apply same mask to all heads
            mask = mask.unsqueeze(1)
        batches = Q.size(0)
        
        # reshape and apply linear projections
        Q, K, V = [
            lin(x).view(batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (Q, K, V))
        ]
        
        # apply attention on all the projected vectors in a batch
        x, self.attn = self.attention(
            Q, K, V, mask=mask
        )
        
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batches, -1, self.h * self.d_k)
        )
        del Q
        del K
        del V
        return self.linears[-1](x)
    
    
    def attention(self, Q: Tensor, K: Tensor, V: Tensor, mask: Union[Tensor,None]=None):
        """
        Scaled dot product attention
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # set to minus infinity all illegal connections
        
        attn = nn.functional.softmax(scores)
        attn = self.dropout(attn)
        return torch.matmul(attn, V), attn 
        
    
def subsequent_mask(size):
    """
    Mask out subsequent positions, i.e. decoder can only use what has already been predicted.
    Direct quote from annotated implementation.
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, heads: int, dropout: float):
        super().__init__()
        self.attn = MultiAttention(heads,d_model,dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sub_1 = AddAndNorm(d_model, dropout)
        self.sub_2 = AddAndNorm(d_model, dropout)
        
    def forward(self, X, mask):
        X = self.sub_1(X, lambda a: self.attn(a, a, a, mask))
        return self.sub_2(X, self.feed_forward(X))
        

class Encoder(nn.Module):
    def __init__(self, N: int, d_model: int, d_ff: int, heads: int, dropout: float):
        super().__init__()
        self.layers = clonelayer(N, EncoderLayer, d_model, d_ff, heads, dropout)
        self.norm = LayerNormaliser(d_model)
    
    def forward(self, X: Tensor, mask):
        """
        Normalise the entire thing at the end, and pass mask through each `EncoderLayer`
        """
        for layer in self.layers:
            X = layer(X, mask)
            
        return self.norm(X)
    
    