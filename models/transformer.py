from math import log, sqrt
from typing import List, Type, Union
from torch import Tensor, nn, tensor
import torch
import copy

# default pytorch is (seq_length, batch_size, data_size)
# alternative is (batch, seq, data)
# thinking I will use batch first as most intuitive to me

##############################################################################################################
##                                                                                                          ##
##                                                                                                          ##
##                                                                                                          ##
##               NB FOR TRAINING: tgt_seq = Y_batch[:, :-1]     (and therefore for mask stuff as well)      ##
##                                                                                                          ##
##               This is because, during training, we want the model to predict the next token in the       ##
##               sequence based on the provided input and the ground truth sequence up to                   ##
##               the second-to-last token.                                                                  ##
##                                                                                                          ##
##                                                                                                          ##
##                                                                                                          ##
##               for loss calculation use   Y_batch[:, 1:]                                                  ##
##                                                                                                          ##
##               This way, you are essentially using the ground truth for the initial token and             ##
##               then using the model's predictions for the subsequent tokens during training               ##
##                                                                                                          ##
##                                                                                                          ##
##############################################################################################################

def clonelayer(N: int, layer: Type[nn.Module], *args, **kwargs):
    '''
    Produces N identical instances of the layer class. 
    args and kwargs are arguments to the constructor of layer. 
    '''
    return nn.ModuleList([layer(*args, **kwargs) for _ in range(N)])


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


class LayerNormaliser(nn.Module):
    '''
    Performs layer normalisation on the input
    '''
    def __init__(self, d_model: int, epsilon=1e-5) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon  # prevent division by zero
        # self.mps = torch.backends.mps.is_available()
        
    def forward(self, X: Tensor):
        mu = torch.mean(X, dim=-1, keepdim=True)
        sigma = torch.std(X, dim=-1, keepdim=True)      
        # sigma, mu = torch.std_mean(X, dim=-1, keepdim=True) 
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
        return X + sublayer(self.layer_norm(X))
    
    
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
        return self.l_2(self.dropout(self.relu(self.l_1(X.to(self.l_1.weight.dtype)))))
    

class TokEmbeddings(nn.Module):
    '''
    Embeddings for the transformer. Can be initialised from a pretrained numpy array. 
    '''
    def __init__(self, d_model: int, vocab_size: int, from_pretrained=None, freeze=True):
        super().__init__()
        self.d_model = d_model
        if from_pretrained is not None:
            assert from_pretrained.shape == (vocab_size, d_model)
            self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(from_pretrained), freeze=freeze)
        else:
            self.embeddings = nn.Embedding(vocab_size, d_model)
        
    def forward(self, X: Tensor):
        return self.embeddings(X) * sqrt(self.d_model)
    

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
        X = X + self.pos_enc[:, :X.size(1)].requires_grad_(False) # this assumes batch first, otherwise use self.pe[:X.size(0)]
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
        
        Q, K, V = [x.to(self.linears[0].weight.dtype) for x in (Q, K, V)]
        
        
        # reshape and apply linear projections
        Q, K, V = [
            lin(x).view(batches, -1, self.heads, self.d_k).transpose(1, 2)
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
            .view(batches, -1, self.heads * self.d_k)
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
            # sc1 = torch.eq(scores, -1e9)
            # # Count the number of True values
            # count1 = torch.sum(sc1).item()
            scores = scores.masked_fill(mask == 0, -1e9) # set to minus infinity all illegal connections
            # sc2 = torch.eq(scores,-1e9)
            # count2 = torch.sum(sc2).item()
            # print("Shape:", scores.shape)
            # print("This many masked: ", count2 - count1)
        
        attn = nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V), attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, heads: int, dropout: float):
        super().__init__()
        self.attn = MultiAttention(heads,d_model,dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sub_1 = AddAndNorm(d_model, dropout)
        self.sub_2 = AddAndNorm(d_model, dropout)
        
    def forward(self, X, mask):
        X = self.sub_1(X, lambda a: self.attn(a, a, a, mask))
        return self.sub_2(X, self.feed_forward)
        

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
    
    
class DecoderLayer(nn.Module):
    def __init__(self, heads: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiAttention(heads, d_model, dropout)
        self.src_attn = MultiAttention(heads, d_model, dropout)
        
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.sub_1 = AddAndNorm(d_model, dropout)
        self.sub_2 = AddAndNorm(d_model, dropout)  
        self.sub_3 = AddAndNorm(d_model, dropout)
        
    def forward(self, X: Tensor, from_encoder: Tensor, encoder_mask, outputs_mask):
        X = self.sub_1(X, lambda x: self.self_attn(x, x, x, outputs_mask))
        X = self.sub_2(X, lambda x: self.src_attn(x, from_encoder, from_encoder, encoder_mask))
        return self.sub_3(X, self.feed_forward)
    

class Decoder(nn.Module):
    def __init__(self, N: int, d_model: int, d_ff: int, heads: int, dropout: float):
        super().__init__()
        self.layers = clonelayer(N, DecoderLayer, heads, d_model, d_ff, dropout)
        self.norm = LayerNormaliser(d_model)
    
    def forward(self, X: Tensor, from_encoder: Tensor, encoder_mask, outputs_mask):
        """
        Normalise the entire thing at the end, and pass mask through each `DecoderLayer`
        """
        for layer in self.layers:
            X = layer(X, from_encoder, encoder_mask, outputs_mask)
            
        return self.norm(X)
    
class OutputGenerator(nn.Module):
    def __init__(self, d_model: int, tgt_vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, X: Tensor, use_non_log_softmax=False):
        X = X.to(self.linear.weight.dtype)
        if use_non_log_softmax:
            return nn.functional.softmax(self.linear(X), dim=-1)
        return nn.functional.log_softmax(self.linear(X), dim=-1)
    
    
class Transformer(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int,
                 N=8, d_model=512, d_ff=2048, heads=8, dropout=0.1, pos_enc_max_len=5000,# hyperparameter defaults from paper
                 input_embeddings=None, freeze_in=True, 
                 output_embeddings=None, freeze_out=True):
        """Initializes internal Module state, shared by both nn.Module and ScriptModule.

        Args:
            in_vocab_size (int): Size of input vocab.
            out_vocab_size (int): Size of output vocab. 
            N (int, optional): Number of cloned layers. Defaults to 8.
            d_model (int, optional): The model's internal dimension. Defaults to 512.
            d_ff (int, optional): The model's convolution dimension. Defaults to 2048 (i.e. 4 * d_model).
            heads (int, optional): Number of heads in the attention layer. Must divide d_model. Defaults to 8.
            dropout (float, optional): Probability of dropout. Defaults to 0.1.
            input_embeddings (NDArray[float32], optional): Numpy array of input embeddings to use. Defaults to None.
            freeze_in (bool, optional): Decides whether to continue learning on pretrained input embeddings. Defaults to True.
            output_embeddings (NDArray[float32], optional): Numpy array of output embeddings to use. Defaults to None.
            freeze_out (bool, optional): Decides whether to continue learning on pretrained output embeddings. Defaults to True.
        """

        super().__init__()
        self.encoder = Encoder(N, d_model, d_ff, heads, dropout)
        self.decoder = Decoder(N, d_model, d_ff, heads, dropout)
        
        pos_enc = PositionalEncoding(d_model, dropout, pos_enc_max_len)
        
        # embeddings should be token embeddings and positional encoding in sequential
        self.input_embeddings = nn.Sequential(
            TokEmbeddings(d_model, in_vocab_size, input_embeddings, freeze_in),
            copy.deepcopy(pos_enc)
        )
        self.output_embeddings = nn.Sequential(
            TokEmbeddings(d_model, out_vocab_size, output_embeddings, freeze_out),
            copy.deepcopy(pos_enc)
        )
        
        self.generator = OutputGenerator(d_model, out_vocab_size)
        
        # initialise all parameters according to xavier uniform
        for n,p in self.named_parameters():
            # skip initialisation of embeddings if using pretrained embeddings
            if n.startswith("input_embeddings") and input_embeddings is not None:
                continue
            if n.startswith("output_embeddings") and output_embeddings is not None:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        
    def forward(self, source: Tensor, target: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        return self.decode(self.encode(source, src_mask), src_mask, target, tgt_mask)
        
    def encode(self, X: Tensor, mask: Tensor):
        return self.encoder(self.input_embeddings(X), mask)

    def decode(self, from_encoder: Tensor, in_mask: Tensor, output: Tensor, out_mask: Tensor) -> Tensor:
        return self.decoder(self.output_embeddings(output), from_encoder, in_mask, out_mask)
    
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        memory = self.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for _ in range(max_len - 1):
            out = self.decode(
                memory, src_mask, ys, self.subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = self.generator(out[:,-1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys