from typing import List, Tuple, Callable, Iterator

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np


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

class Encoder(nn.Module):
    """Encodes the static states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)

class Decoder(nn.Module):
    """Decodes the hidden states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)

class Actor(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    hidden_size: int
        Defines the number of units in the hidden layer for all static
        and decoder output units.
    num_layers: int, optional
        Specifies the number of hidden layers to use in the decoder RNN, by default 1
    dropout: float, optional
        Defines the dropout rate for the decoder, by default 0
    """

    def __init__(self, state_size, hidden_size, device, num_layers=1, dropout=0.):
        super(Actor, self).__init__()
        # Define the encoder & decoder models
        self.state_encoder = Encoder(state_size, hidden_size)
        self.decoder = Decoder(state_size, hidden_size)
        self.pointer = Pointer(hidden_size, device, num_layers, dropout)
        self.device = device

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1,state_size, 1), requires_grad=True, device=self.device)

    def forward(self, state, intersections, previous_action=None, last_hh=None):
        """
        Parameters
        ----------
        state: Array of size (num_intersections)
            Mask of the intersections that are selected for the RSU network.
        previous_action: Array of size (num_intersections)
            One hot array defining the last intersection that was selected.
        intersections: Array of size (feats,num_intersections)
            Defines the features for all intersections.
        last_hh: Array of size (num_hidden)
            Defines the last hidden state for the RNN
        """
        action = torch.ones(*state.shape,dtype=torch.int)
        encoder_input = intersections
        # encoder_input = torch.gather(intersections, 2, torch.from_numpy(np.where(state.detach().numpy()==1)[1]).expand(*intersections.shape[0:2],-1)).detach()

        if previous_action is None: # If there was not a previous action set encoder input to 0
            decoder_input = self.x0
        else:
            decoder_input = torch.gather(intersections, 2, torch.argmin(previous_action).expand(*intersections.shape[0:2],1)).detach()
        # Applies encoding to all potential intersections
        state_hidden = self.state_encoder(encoder_input)
        decoder_hidden = self.decoder(decoder_input)
        probs, current_hh = self.pointer(state_hidden, decoder_hidden, last_hh)

        probs = F.softmax(probs + state.log(), dim=1)
        # When training, sample the next step according to its probability.
        # During testing, we can take the greedy approach and choose highest
        if self.training:
            m = torch.distributions.Categorical(probs)

            ptr = m.sample()
            while not torch.gather(state, 1, ptr.data.unsqueeze(1)).byte().all():
                ptr = m.sample()
            logp = m.log_prob(ptr)
        else:
            prob, ptr = torch.max(probs,1)  # Greedy
            logp = prob.log()

        RSU_logp = logp.unsqueeze(1)
        RSU_idx = ptr.data.unsqueeze(1)

        
        action[0,RSU_idx] = 0

        return action, RSU_logp, current_hh

class Critic(nn.Module):
    """Estimates the problem complexity.
    """

    def __init__(self, static_size, hidden_size):
        super(Critic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, state, intersections):

        encoder_input = intersections
        # encoder_input = torch.gather(intersections, 2, torch.from_numpy(np.where(state.detach().numpy()==1)[1]).expand(*intersections.shape[0:2],-1)).detach()

        hidden = self.static_encoder(encoder_input)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output

class Actor_Critic(nn.Module):
    def __init__(self, actor_net: nn.Module, critic_net: nn.Module, agent, device = 'cpu') -> None:
        super().__init__()
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.agent = agent

        self.previous_action = None
        self.previous_hh = None
        self.device = device
    
    def forward(self,state: torch.Tensor) -> Tuple:
        state = state.to(self.device)
        intersections = self.agent.intersections

        # if len(self.batch_actions) == 0: previous_action = None # If there was no previous action
        # else: previous_action = self.batch_actions[-1]

        # if len(self.last_hhs) == 0: last_hh = None
        # else: last_hh = self.last_hhs[-1]

        action, logp, current_hh = self.actor_net(state, intersections, self.previous_action, self.previous_hh)
        critic_reward = self.critic_net(state, intersections).view(-1)

        # Save for next run of the actor
        self.previous_action = action.detach()
        self.previous_hh = current_hh.detach()

        return action, logp, critic_reward

    def reset(self) -> None:
        self.previous_action = None
        self.previous_hh = None
