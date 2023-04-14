from cmath import isnan
from typing import List, Tuple, Callable, Iterator

import torch
from torch import Tensor, nn
import torch.nn.functional as F

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

    def forward(self,items):
        return self.transformer(items)

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
            x_decoder: (B, Nd, C)
            x_encoder: (B, Ne, C)
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
        # print(encoder_transform.shape)
        # print(decoder_transform.shape)
        prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        # print(prod.shape)
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
    def __init__(self, 
                 num_features = 4,
                 nhead = 4,
                 W = [.5,.5],
                 c_embed = 256,
                 n_layers = 1
                 ):
        super(Actor,self).__init__()
        self.c_embed = c_embed
        self.embedding = Embedding(num_features,c_embed)
        self.encoder = Encoder(c_embed,nhead,n_layers)
        self.decoder = Decoder(c_embed,nhead,n_layers)
        self.pointer = PointerNetwork(c_embed)
        self.W = W

    def forward(self, items, budget):

        mask = np.ones(shape=items.shape[1]+1) # 1 is added for "No item" selection
        items = torch.hstack((torch.zeros(size = (1,1,2)),items))
        # print("items",items)
        # print("mask",mask)
        # print("Input shape",items.shape)

        # print("Embedding:\n")
        # for p in self.embedding.parameters():
        #     if p.requires_grad:
        #         print(p)
        # print("Encoder:\n")
        # for p in self.encoder.parameters():
        #     if p.requires_grad:
        #         print(p)
        # print("Decoder:\n")
        # for p in self.decoder.parameters():
        #     if p.requires_grad:
        #         print(p)

        embedded_state = self.embedding(items)
        # print("embedded_state",embedded_state.shape)

        encoder_state = self.encoder(embedded_state)
        # print("encoder_state",encoder_state.shape)
        decoder_input = encoder_state[:,:1,:]
        # print("decoder input",decoder_input.shape)

        if np.isnan(decoder_input.detach().numpy().sum()): quit()

        q_values = []
        masked_argmaxs = []
        for i in range(items.shape[1]-1):
            decoder_state = self.decoder(decoder_input,encoder_state)
            # print("decoder_state",decoder_state)

            q_value = self.pointer(decoder_state,encoder_state)
            # print("Pointer output",q_value.shape)
            print(q_value)
            _, masked_argmax = self.masked_max(q_value,mask, dim=-1)
            print(masked_argmax)
            print('\n')
            # print("masked argmax",masked_argmax.shape)
            q_values.append(q_value[:, -1, :])
            new_maxes = masked_argmax[:, -1]
            # print("mask",mask)
            # print("New maxes",new_maxes)
            mask[new_maxes] = False
            # mask = mask.unsqueeze(1).expand(-1, q_value.shape[1], -1)
            masked_argmaxs.append(new_maxes)
            # print("masked argmaxes array",masked_argmaxs)
            # print('\n')
            mask[0] = 1
            if (1-mask).sum() == len(mask)-1:
                break
            if mask.all():
                break
        
            next_indices = torch.stack(masked_argmaxs, dim=1).unsqueeze(-1).expand(items.shape[0], -1, self.c_embed)
            decoder_input = torch.cat((encoder_state[:,:1,:], torch.gather(encoder_state, dim=1, index=next_indices)), dim=1)
        q_values = torch.stack(q_values, dim=1)
        masked_argmaxs = torch.stack(masked_argmaxs, dim=1)
        # print(q_values)
        return q_values, masked_argmaxs
    
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
        # print("x",x)
        # print("mask",mask)
        x_replaced = x*Tensor(mask)
        x_replaced[x_replaced == 0] = -3e37
        # print("x_replaced",x_replaced)
        max_value, max_index = x_replaced.max(dim=dim, keepdim=keepdim)
        # print("max val",max_value)
        # print("max index",max_index)
        return max_value, max_index

class Critic(nn.Module):
    def __init__(self,num_features = 3,
                      c_embed = 16 ):
        super(Critic,self).__init__()
        self.embedding = Embedding(num_features,c_embed)
        self.network = nn.Sequential(nn.Linear(c_embed,16),
                                     nn.ReLU(),
                                     nn.Linear(16,32),
                                     nn.ReLU(),
                                     nn.Linear(32,64),
                                     nn.ReLU(),
                                     nn.Linear(64,32),
                                     nn.ReLU(),
                                     nn.Linear(32,16),
                                     nn.ReLU(),
                                     nn.Linear(16,1))
        

    def forward(self,items):
        embedded_state = self.embedding(items)

        rewards = []
        for sample in embedded_state:
            reward = self.network(sample)
            rewards.append(reward)
        rewards = torch.stack(rewards,dim=0)

        return rewards[:,:,0]

class Actor_Critic(nn.Module):
    def __init__(self, num_features: int,
                       nhead: int,
                       W = [.5,.5],
                       n_layers = 1) -> None:
        super(Actor_Critic,self).__init__()
        self.actor = Actor(num_features,nhead,W = W, n_layers = n_layers)
        self.critic = Critic(num_features)
    
    def forward(self,items: torch.Tensor, budget: int):
        log_pointer_scores, pointer_argmaxs = self.actor(items, budget)
        pack_idx = pointer_argmaxs[pointer_argmaxs>0]-1
        rsu_network = items[:,pack_idx,:]
        # critic_reward = self.critic(rsu_network)
        critic_reward = torch.tensor(0)

        return log_pointer_scores, pointer_argmaxs, pack_idx, critic_reward

    def reset(self) -> None:
        self.previous_action = None
        self.previous_hh = None
