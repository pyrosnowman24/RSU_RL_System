from turtle import forward
from typing import List, Tuple, Callable, Iterator

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

import numpy as np

class Embedding(nn.Module):
    def __init__(self,c_inputs,c_embed):
        super(Embedding,self).__init__()
        self.embedding_layer = nn.Linear(c_inputs, c_embed, bias=False)

    def forward(self,data):
        return self.embedding_layer(data)

class Encoder(nn.Module):
    def __init__(self,num_features,nhead,n_layers = 1):
        super(Encoder,self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(num_features,nhead,batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer,n_layers)

    def forward(self,intersections):
        return self.transformer(intersections)

class Decoder(nn.Module):
    def __init__(self,num_features,nhead,n_layers = 1):
        super(Decoder,self).__init__()
        transformer_layer = nn.TransformerDecoderLayer(num_features,nhead,batch_first=True)
        self.transformer = nn.TransformerDecoder(transformer_layer,n_layers)

    def forward(self,rsu_intersections,encoder_state):
        return self.transformer(rsu_intersections,encoder_state)

class PointerNetwork(nn.Module):
    """
    From "Pointer Networks" by Vinyals et al. (2017)
    Adapted from pointer-networks-pytorch by ast0414:
    https://github.com/ast0414/pointer-networks-pytorch
    Args:
    n_hidden: The number of features to expect in the inputs.
    """

    def __init__(self, n_hidden: int):
        super().__init__()
        self.n_hidden = n_hidden
        self.w1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.w2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.v = nn.Linear(n_hidden, 1, bias=False)

    def forward(self,
        x_decoder: torch.Tensor,
        x_encoder: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_decoder: Encoding over the output tokens.
            x_encoder: Encoding over the input tokens.
        Shape:
            x_decoder: (B, Ne, C)
            x_encoder: (B, Nd, C)
        """

        # print("\n x encoded shape",x_encoder.shape)
        # print("x encoded",x_encoder)
        # print("x decoder shape",x_decoder.shape)
        # print("x decoder",x_decoder)

        # (B, Nd, Ne, C) <- (B, Ne, C)
        encoder_transform = self.w1(x_encoder).unsqueeze(1).expand(
            -1, x_decoder.shape[1], -1, -1)
        # print("encoder transform output",encoder_transform.shape)
        # print("encoder transform",encoder_transform)

        # (B, Nd, 1, C) <- (B, Nd, C)
        decoder_transform = self.w2(x_decoder).unsqueeze(2)
        # print("decoder transform shape",decoder_transform.shape)
        # print("decoder transform",decoder_transform)

        # (B, Nd, Ne) <- (B, Nd, Ne, C), (B, Nd, 1, C)
        prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        # print("pointer output",prod.shape)
        # print("pointer output",prod)

        # (B, Nd, Ne) <- (B, Nd, Ne)
        log_score = self.log_softmax(x = prod, dim=-1)

        # print("Log score shape", log_score.shape)
        # print("Log score",log_score,'\n')

        return log_score

    def log_softmax(self,
        x: torch.Tensor,
        dim: int = -1,
        ) -> torch.Tensor:
        """
        Apply softmax to x with masking.

        Adapted from allennlp by allenai:
            https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

        Args:
            x - Tensor of arbitrary shape to apply softmax over.
            dim - Dimension over which to apply operation.
        Outputs:
            Tensor with same dimensions as x.
        """
        return torch.nn.functional.log_softmax(x, dim=dim)

class Actor(nn.Module):
    def __init__(self,num_features = 3,
                      nhead = 4,
                      W = [.5,.5],
                      c_embed = 16,
                      n_layers = 1):
        super(Actor,self).__init__()
        self.c_embed = c_embed
        self.embedding = Embedding(nhead,c_embed)
        self.encoder = Encoder(c_embed,nhead,n_layers)
        self.decoder = Decoder(c_embed,nhead,n_layers)
        self.pointer = PointerNetwork(c_embed)
        self.W = W

    def forward(self,intersections,mask):
        # print("Input shape",intersections.shape)

        embedded_state = self.embedding(intersections)
        # print("embedded_state",embedded_state.shape)

        encoder_state = self.encoder(embedded_state)
        # print("encoder_state",encoder_state.shape)

        decoder_input = encoder_state[:,:1,:]
        # print("decoder input shape",decoder_input.shape)

        log_pointer_scores = []
        masked_argmaxs = []
        for i in range(intersections.shape[1]):
            if i <= 1: mask[:,0] = False
            decoder_state = self.decoder(decoder_input,encoder_state)
            # print("decoder_state",decoder_state.shape)

            log_pointer_score = self.pointer(decoder_state,encoder_state)
            # print("Pointer output shape",log_pointer_score.shape)
            _, masked_argmax = self.masked_max(log_pointer_score,mask, dim=-1)
            # print("masked argmax",masked_argmax)

            log_pointer_scores.append(log_pointer_score[:, -1, :])
            new_maxes = masked_argmax[:, -1]
            # print("New maxes",new_maxes)
            mask[0,new_maxes] = False
            mask[:,0] = True # This is the choice that no RSU should be placed. 
            # mask = mask.unsqueeze(1).expand(-1, log_pointer_score.shape[1], -1)
            masked_argmaxs.append(new_maxes)
            # print("masked argmaxes array",masked_argmaxs)
            # print('\n')
            next_indices = torch.stack(masked_argmaxs, dim=1).unsqueeze(-1).expand(intersections.shape[0], -1, self.c_embed)
            decoder_input = torch.cat((encoder_state[:,:1,:], torch.gather(encoder_state, dim=1, index=next_indices)), dim=1)

        log_pointer_scores = torch.stack(log_pointer_scores, dim=1)
        masked_argmaxs = torch.stack(masked_argmaxs, dim=1)
        
        return log_pointer_scores, masked_argmaxs

    def masked_max(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
        keepdim: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply max to x with masking.

        Adapted from allennlp by allenai:
            https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

        Args:
            x - Tensor of arbitrary shape to apply max over.
            mask - Binary mask of same shape as x where "False" indicates elements
            to disregard from operation.
            dim - Dimension over which to apply operation.
            keepdim - If True, keeps dimension dim after operation.
        Outputs:
            A ``torch.Tensor`` of including the maximum values.
        """
        x_replaced = x.masked_fill(~mask, -3e37)
        # print(x_replaced)
        max_value, max_index = x_replaced.max(dim=dim, keepdim=keepdim)
        # print(max_value)
        # print(max_index)
        return max_value, max_index

class Policy_Gradient(nn.Module):
    def __init__(self, num_features: int,
                       nhead: int,
                       W = [.5,.5],
                       n_layers = 1) -> None:
        super(Policy_Gradient,self).__init__()
        self.actor = Actor(num_features,nhead,W = W, n_layers = n_layers)

    def forward(self,intersections: torch.Tensor,
                     intersection_idx: torch.Tensor,
                     rsu_network_idx: torch.Tensor,
                     mask: torch.Tensor):

        log_pointer_scores, pointer_argmaxs = self.actor(intersections,mask)
        return log_pointer_scores, pointer_argmaxs


