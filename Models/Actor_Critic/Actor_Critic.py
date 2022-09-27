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

    def forward(self,intersections):
        return self.transformer(intersections)

class Decoder(nn.Module):
    def __init__(self,num_features,nhead,n_layers = 1):
        super(Decoder,self).__init__()
        transformer_layer = nn.TransformerDecoderLayer(num_features,nhead,batch_first=True)
        self.transformer = nn.TransformerDecoder(transformer_layer,n_layers)

    def forward(self,rsu_intersections,encoder_state):
        return self.transformer(rsu_intersections,encoder_state)

class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size, device):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size), device=device, requires_grad=True))

    def forward(self, static_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns

class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, device, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size), device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size, device)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            current_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, current_hh

class Actor(nn.Module):
    def __init__(self,num_features = 3,
                      nhead = 4,
                      W = [.5,.5],
                      c_embed = 16,
                      n_layers = 1):
        super(Actor,self).__init__()
        self.c_embed = c_embed
        self.embedding = Embedding(num_features,c_embed)
        self.encoder = Encoder(c_embed,nhead,n_layers)
        self.decoder = Decoder(c_embed,nhead,n_layers)
        self.pointer = Pointer(c_embed)
        self.W = W

    def forward(self,intersections,mask):
        # print("Input shape",intersections.shape)

        embedded_state = self.embedding(intersections)
        # print("embedded_state",embedded_state.shape)

        encoder_state = self.encoder(embedded_state)
        # print("encoder_state",encoder_state.shape)

        decoder_input = encoder_state[:,:1,:]
        # print("decoder input shape",decoder_input.shape)

        q_values = []
        masked_argmaxs = []
        for i in range(intersections.shape[1]):
            if i == 0: mask[:,0] = False
            decoder_state = self.decoder(decoder_input,encoder_state)
            # print("decoder_state",decoder_state.shape)

            q_value = self.pointer(decoder_state,encoder_state)
            # print("Pointer output shape",q_value.shape)
            _, masked_argmax = self.masked_max(q_value,mask, dim=-1)
            # print("masked argmax",masked_argmax)

            q_values.append(q_value[:, -1, :])
            new_maxes = masked_argmax[:, -1]
            # print(mask)
            # print(new_maxes)
            # print("New maxes",new_maxes)
            mask[0,new_maxes] = False
            mask[:,0] = True # This is the choice that no RSU should be placed. 
            # mask = mask.unsqueeze(1).expand(-1, q_value.shape[1], -1)
            masked_argmaxs.append(new_maxes)
            # print("masked argmaxes array",masked_argmaxs)
            # print('\n')
            if (~mask).all():
                break

            next_indices = torch.stack(masked_argmaxs, dim=1).unsqueeze(-1).expand(intersections.shape[0], -1, self.c_embed)
            decoder_input = torch.cat((encoder_state[:,:1,:], torch.gather(encoder_state, dim=1, index=next_indices)), dim=1)

        q_values = torch.stack(q_values, dim=1)
        masked_argmaxs = torch.stack(masked_argmaxs, dim=1)
        # print(masked_argmax)
        # quit()
        return q_values, masked_argmaxs, mask

class Critic(nn.Module):
    def __init__(self,num_features = 3,
                      c_embed = 16 ):
        super(Critic,self).__init__()
        self.embedding = Embedding(num_features,c_embed)
        self.network = nn.Sequential(nn.Linear(c_embed,16),
                                     nn.ReLU(),
                                     nn.Linear(16,32),
                                     nn.ReLU(),
                                     nn.Linear(32,16),
                                     nn.ReLU(),
                                     nn.Linear(16,1))
        

    def forward(self,intersections):
        embedded_state = self.embedding(intersections)

        rewards = []
        for sample in embedded_state:
            reward = self.network(sample)
            rewards.append(reward)
        rewards = torch.stack(rewards,dim=1)

        return rewards

class Actor_Critic(nn.Module):
    def __init__(self, num_features: int,
                       nhead: int,
                       W = [.5,.5],
                       n_layers = 1) -> None:
        super(Actor_Critic,self).__init__()
        self.actor = Actor(num_features,nhead,W = W, n_layers = n_layers)
        self.critic = Critic(num_features)
    
    def forward(self,intersections: torch.Tensor,
                     mask: torch.Tensor):
        
        log_pointer_scores, pointer_argmaxs = self.actor(intersections,mask)
        rsu_idx = pointer_argmaxs[pointer_argmaxs>0]-1
        rsu_network = intersections[rsu_idx,:]
        critic_reward = self.critic(rsu_network)

        return log_pointer_scores, pointer_argmaxs, rsu_network, critic_reward

    def reset(self) -> None:
        self.previous_action = None
        self.previous_hh = None
