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
    def __init__(self, num_features, hidden_size: int):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
    
    def forward(self, x: torch.Tensor):
        # x: (BATCH, ARRAY_LEN, 1)
        return self.lstm(x)

class Decoder(nn.Module):
    def __init__(self, 
               hidden_size: int,
               attention_units: int = 10):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size + 2, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size, attention_units)

    def forward(self, 
              x: torch.Tensor, 
              hidden: Tuple[torch.Tensor], 
              encoder_out: torch.Tensor):
        # x: (BATCH, 1, 1) 
        # hidden: (1, BATCH, HIDDEN_SIZE)
        # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
        # For a better understanding about hidden shapes read: https://pytorch.org/docs/stable/nn.html#lstm
        
        # Get hidden states (not cell states) 
        # from the first and unique LSTM layer 
        ht = hidden[0][0]  # ht: (BATCH, HIDDEN_SIZE)

        # di: Attention aware hidden state -> (BATCH, HIDDEN_SIZE)
        # att_w: Not 'softmaxed', torch will take care of it -> (BATCH, ARRAY_LEN)
        di, att_w = self.attention(encoder_out, ht)
        
        # Append attention aware hidden state to our input
        # x: (BATCH, 1, 1 + HIDDEN_SIZE)
        x = torch.cat([di.unsqueeze(1), x], dim=2)
        
        # Generate the hidden state for next timestep
        _, hidden = self.lstm(x, hidden)
        return hidden, att_w

class Attention(nn.Module):
    """
    From "Pointer Networks" by Vinyals et al. (2017)
    Adapted from pointer-networks-pytorch by ast0414:
    https://github.com/ast0414/pointer-networks-pytorch
    Args:
    n_hidden: The number of features to expect in the inputs.
    """

    def __init__(self, hidden_size, attention_units):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, attention_units, bias=False)
        self.W2 = nn.Linear(hidden_size, attention_units, bias=False)
        self.V =  nn.Linear(attention_units, 1, bias=False)

    def forward(self, 
                encoder_out: torch.Tensor, 
                decoder_hidden: torch.Tensor):
        # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
        # decoder_hidden: (BATCH, HIDDEN_SIZE)

        # Add time axis to decoder hidden state
        # in order to make operations compatible with encoder_out
        # decoder_hidden_time: (BATCH, 1, HIDDEN_SIZE)
        decoder_hidden_time = decoder_hidden.unsqueeze(1)

        # uj: (BATCH, ARRAY_LEN, NUM_VALUES)
        # Note: we can add the both linear outputs thanks to broadcasting
        uj = self.W1(encoder_out) + self.W2(decoder_hidden_time)
        uj = torch.tanh(uj)

        # uj: (BATCH, ARRAY_LEN, 1)
        uj = self.V(uj)

        # Attention mask over inputs
        # aj: (BATCH, ARRAY_LEN, 1)
        aj = F.softmax(uj, dim=1)

        # di_prime: (BATCH, HIDDEN_SIZE)
        di_prime = aj * encoder_out
        di_prime = di_prime.sum(1)
        # print(uj)
        
        return di_prime, uj.squeeze(-1)

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
                 hidden_size = 256,
                 embedding_size = 16,
                 attention_units = 256
                 ):
        super(Actor,self).__init__()
        self.embedding = Embedding(num_features,embedding_size)
        self.encoder = Encoder(embedding_size,hidden_size)
        self.decoder = Decoder(hidden_size,attention_units)

    def forward(self, items, budget):

        mask = np.ones(shape=items.shape[1]+1) # 1 is added for "No item" selection
        mask[0]=0
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

        out, hs = self.encoder(embedded_state)
        
        # Save probabilities at each timestep
        # outputs: (ARRAY_LEN, ITEMS)
        probabilities = None

        # Save outputs at each timestep
        # outputs: (ARRAY_LEN)
        outputs = torch.zeros(items.shape[1], dtype=torch.long)
        
        # First decoder input is always 0
        # dec_in: (BATCH, 1, 1)
        dec_in = torch.zeros(out.size(0), 1, 2, dtype=torch.float)

        for i in range(items.shape[1]):
            hs, att_w = self.decoder(dec_in, hs, out)
            if i == 0:
                probabilities = self.masked_log_softmax(att_w, mask)
            else:
                probabilities = torch.vstack((probabilities, self.masked_log_softmax(att_w, mask)))
            # print(probabilities)
            predictions = self.masked_softmax(att_w, mask)
            mask[predictions] = 0
            mask[0] = 1
            idx = predictions
            dec_in = torch.stack([items[b, idx[b].item()] for b in range(items.size(0))])
            dec_in = dec_in.view(out.size(0), 1, 2).type(torch.float)
            outputs[i] = predictions
        return probabilities, outputs
    
    def masked_softmax(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
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
        # print("mask",mask.shape)
        x_replaced = x*Tensor(mask)
        x_replaced[x_replaced == 0] = -3e37
        # print("x_replaced",x_replaced)
        predictions = F.softmax(x_replaced, dim=1).argmax(1)
        # print("max val",max_value)
        # print("max index",max_index)
        return predictions
    
    def masked_log_softmax(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
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
        # """
        # print("x",x)
        # print("mask",mask)
        x_replaced = x*Tensor(mask)
        x_replaced[x_replaced == 0] = -3e-37
        predictions = F.log_softmax(x_replaced, dim=-1)
        return predictions

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
                       hidden_size: int,
                       embedding_size = 16,
                       attention_units = 10) -> None:
        super(Actor_Critic,self).__init__()
        self.actor = Actor(num_features,hidden_size,embedding_size, attention_units)
        # self.critic = Critic(num_features)
    
    def forward(self,items: torch.Tensor, budget: int):
        log_pointer_scores, pointer_argmaxs = self.actor(items, budget)
        pack_idx = pointer_argmaxs[pointer_argmaxs>0]-1
        # rsu_network = items[:,pack_idx,:]
        # critic_reward = self.critic(rsu_network)
        critic_reward = torch.tensor(0)

        return log_pointer_scores, pointer_argmaxs, pack_idx, critic_reward

    def reset(self) -> None:
        self.previous_action = None
        self.previous_hh = None
