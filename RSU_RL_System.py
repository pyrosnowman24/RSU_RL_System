from typing import List, Tuple, Callable, Iterator
from collections import OrderedDict, deque, namedtuple

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import IterableDataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import argparse

# This is a advantage actor-critic model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(LightningModule):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

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

class Pointer(LightningModule):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh

class Encoder(LightningModule):
    """Encodes the static states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)

class Decoder(LightningModule):
    """Decodes the hidden states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)

class Actor(LightningModule):
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

    def __init__(self, state_size, hidden_size, num_layers=1, dropout=0.):
        super(Actor, self).__init__()


        # Define the encoder & decoder models
        self.state_encoder = Encoder(state_size, hidden_size)
        self.decoder = Decoder(state_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1,state_size, 1), requires_grad=True, device=device)

    def forward(self, state, previous_action, intersections, last_hh=None):
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
        mask = torch.ones(*state.shape,dtype=torch.int)
        if previous_action is None: # If there was not a previous action set encoder input to 0
            decoder_input = self.x0
        else:
            decoder_input = torch.gather(intersections, 2, torch.argmin(previous_action).expand(*intersections.shape[0:2],1)).detach()
        # Applies encoding to all intersection data, becasue they are constant it only needs to be done once
        state_hidden = self.state_encoder(intersections)
        # Below was the former loop to create the entire RSU network

        decoder_hidden = self.decoder(decoder_input)

        probs, last_hh = self.pointer(state_hidden, decoder_hidden, last_hh)
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

        
        mask[0,RSU_idx] = 0

        return mask, RSU_logp, last_hh

class Critic(LightningModule):
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
    def forward(self, static, intersections):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(intersections)

        hidden = static_hidden

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output

class Agent:
    def __init__(self):
        intersections = np.array(((1.0,.4,.8,.9,.3),(1.0,.7,.2,.6,.4)))
        self.intersections = torch.tensor(np.reshape(intersections,(1,*intersections.shape)),dtype=torch.float32)

    def simulation_step(self,rsu_indices,w1,w2):
        "Runs a small batch of the simulation"
        reward = 300
        return reward
    
    def reset(self):
        "Resets the simulation environment"
        return torch.ones(1,self.intersections.shape[2],dtype=torch.int)

class ExperienceSourceDataset(IterableDataset):
    """Basic experience source dataset.
    Takes a generate_batch function that returns an iterator. The logic for the experience source and how the batch is
    generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch()
        return iterator

class DRL_System(LightningModule):
    def __init__(self, train_size: int = 120000,
                       valid_size: int = 1000, 
                       hidden_size: int = 128, 
                       num_layers: int = 1, 
                       dropout: float = 0.1, 
                       actor_lr: float = 5e-4,
                       critic_lr: float = 5e-4, 
                       batch_size: int = 200,
                       state_size: int = 2,
                       w1: int = 1, 
                       w2: int = 0):
        """DRL model for solving RSU placement problem

        Args:
            train_size (int, optional): _description_. Defaults to 120000.
            valid_size (int, optional): _description_. Defaults to 1000.
            hidden_size (int, optional): _description_. Defaults to 128.
            num_layers (int, optional): _description_. Defaults to 1.
            dropout (float, optional): _description_. Defaults to 0.1.
            actor_lr (float, optional): _description_. Defaults to 5e-4.
            critic_lr (float, optional): _description_. Defaults to 5e-4.
            batch_size (int, optional): _description_. Defaults to 200.
            state_size (int, optional): _description_. Defaults to 4.
            checkpoint (str, optional): _description_. Defaults to None.
            w1 (int, optional): _description_. Defaults to 1.
            w2 (int, optional): _description_. Defaults to 0.
        """

        super(DRL_System,self).__init__()
        self.train_size = train_size
        self.valid_size = valid_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.state_size = state_size
        self.w1 = w1
        self.w2 = w2

        self.actor = Actor(state_size,
                           self.hidden_size,
                           self.num_layers,
                           self.dropout).to(device)

        self.critic = Critic(state_size,
                             self.hidden_size).to(device)

        self.agent = Agent()

        self.episode_rewards = []
        self.critic_rewards = []
        self.batch_actions = []
        self.avg_rewards = 0
        self.logps = []
        self.total_rewards = []
        self.batch_states = []
        self.last_hhs = []
        self.done_episodes = 0
        self.state = self.agent.reset()
        self.done = False

    def model_loss(self,rewards,critic_ests,logps):
        """Function to calculate the losses for the actor and critic.

        Args:
            rewards (float): The actual rewards for the RSU network.
            critic_est (float): The estimate of the rewards for the RSU network from the critic.
            logps (float): Log probabilities for each of the selected RSUs in the network.

        Returns:
            _type_: _description_
        """
        advantages = torch.sub(rewards,critic_ests)
        advantage = torch.sum(advantages) # Not sure whether to use SUM or MEAN
        actor_loss = torch.mean(advantage.detach() * logps.sum(dim=1))
        critic_loss = torch.mean(advantage ** 2)
        return actor_loss, critic_loss

    def forward(self,x: Tensor) -> Tensor:
        # This must be redone based on the new actor call method, it should still return an RSU network
        rsu_indicies, _ = self.actor(x)
        return rsu_indicies

    def training_step(self,batch:Tuple[Tensor,Tensor],batch_idx,optimizer_idx) -> OrderedDict:
        states, actions, logps, rewards, critic_rewards = batch

        actor_loss, critic_loss = self.model_loss(rewards,critic_rewards,logps)


        if optimizer_idx == 0: loss = actor_loss
        if optimizer_idx == 1: loss = critic_loss

        log = {
            "episodes": self.done_episodes,
            "reward": self.total_rewards[-1],
            "train_loss": loss,
            "avg_reward": self.avg_rewards,
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})
        
    def configure_optimizers(self) -> List[Optimizer]:
        """Initializes Adam optimizers for actor and critic.

        Returns:
            List[Optimizer]: Optimizers for actor and critic.
        """
        actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)
        return [actor_optim, critic_optim]

    def train_batch(self,) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.
        Returns:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """
        """
        States: Mask of 0 and 1 determining which RSUs are selected for the RSU network, 0 is selected and 1 is not selected
        Action: One cold mask determining which RSU is selected to be placed into RSU network
        Logp: Log probability of selected RSU
        Reward: reward from simulation for RSU network
        Critic Reward: Estimation of the reward by the Critic
        Next State: np.bitwise_and of States and Action (Adds selected RSU to mask of RSU network)
        """
        while True:
            intersections = self.agent.intersections

            if len(self.batch_actions) == 0: previous_action = None # If there was no previous action
            else: previous_action = self.batch_actions[-1]

            if len(self.last_hhs) == 0: last_hh = None
            else: last_hh = self.last_hhs[-1]
            action, logp, current_hh = self.actor(self.state,previous_action,intersections,last_hh = last_hh)
            next_state = torch.bitwise_and(self.state,action)
            reward = self.agent.simulation_step(next_state,self.w1,self.w2)

            critic_reward = self.critic(next_state, intersections).view(-1)

            self.episode_rewards.append(reward)
            self.critic_rewards.append(critic_reward)
            self.logps.append(logp)
            self.batch_actions.append(action)
            self.batch_states.append(self.state)
            self.state = next_state
            self.last_hhs.append(current_hh)

            if torch.sum(next_state) == 0: # If all intersections have been selected, then the simulation should end. 
                self.done = True

            if self.done:
                self.done_episodes += 1
                self.done = False
                self.state = self.agent.reset()
                self.total_rewards.append(np.sum(self.episode_rewards))
                self.avg_rewards = float(np.mean(self.total_rewards))

                for idx in range(len(self.batch_actions)):
                    yield self.batch_states[idx], self.batch_actions[idx], self.logps[idx], self.episode_rewards[idx], self.critic_rewards[idx]

                self.batch_states = []
                self.batch_actions = []
                self.logps = []
                self.episode_rewards = []
                self.critic_rewards = []
                self.last_hhs = []

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

if __name__ == '__main__':
    num_nodes = 100
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=120000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)
    args = parser.parse_args()
    T = 100
    w2_list = np.arange(T+1)/T
    w1_list = 1-w2_list
    model = DRL_System(**args.__dict__)
    trainer = Trainer()
    trainer.fit(model)
