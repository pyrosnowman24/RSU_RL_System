from typing import List, Tuple
from collections import OrderedDict, deque, namedtuple
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import IterableDataset
import numpy as np

# This is a advantage actor-critic model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(LightningModule):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

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

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh

class Encoder(LightningModule):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
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
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function, optional
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element, by default None
    mask_fn: function, optional
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever, by default None
    num_layers: int, optional
        Specifies the number of hidden layers to use in the decoder RNN, by default 1
    dropout: float, optional
        Defines the dropout rate for the decoder, by default 0
    """

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(Actor, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static.
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder.
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """

        batch_size, input_size, sequence_size = static.size()

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        for _ in range(max_steps):

            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            decoder_hidden = self.decoder(decoder_input)

            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            probs = F.softmax(probs + mask.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot 
                # in these cases, and logp := 0
                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            decoder_input = torch.gather(static, 2,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp

class Critic(LightningModule):
    """Estimates the problem complexity.
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(Critic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output

class Agent:
    def __init__(self):
        self.hi = "hi"

    def simulation_step(self,rsu_indices):
        print("hi")

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)

class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )

class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

class DRL_System(LightningModule):
    def __init__(self, num_nodes: int = 100, 
                       train_size: int = 120000,
                       valid_size: int = 1000, 
                       hidden_size: int = 128, 
                       num_layers: int = 1, 
                       dropout: float = 0.1, 
                       seed: int = 12345, 
                       actor_lr: float = 5e-4,
                       critic_lr: float = 5e-4, 
                       batch_size: int = 200,
                       static_size: int = 4,
                       dynamic_size: int = 1,
                       checkpoint: str = None,
                       replay_buffer_size: int = 50,
                       episode_length:int = 200,
                       w1: int = 1, 
                       w2: int = 0):
        """DRL model for solving RSU placement problem

        Args:
            num_nodes (int, optional): _description_. Defaults to 100.
            train_size (int, optional): _description_. Defaults to 120000.
            valid_size (int, optional): _description_. Defaults to 1000.
            hidden_size (int, optional): _description_. Defaults to 128.
            num_layers (int, optional): _description_. Defaults to 1.
            dropout (float, optional): _description_. Defaults to 0.1.
            seed (int, optional): _description_. Defaults to 12345.
            actor_lr (float, optional): _description_. Defaults to 5e-4.
            critic_lr (float, optional): _description_. Defaults to 5e-4.
            batch_size (int, optional): _description_. Defaults to 200.
            static_size (int, optional): _description_. Defaults to 4.
            dynamic_size (int, optional): _description_. Defaults to 1.
            checkpoint (str, optional): _description_. Defaults to None.
            replay_buffer_size (int, optional): _description_. Defaults to 50.
            episode_length (int, optional): _description_. Defaults to 200.
            w1 (int, optional): _description_. Defaults to 1.
            w2 (int, optional): _description_. Defaults to 0.
        """

        super(DRL_System,self).__init__()
        self.num_nodes = num_nodes
        self.train_size = train_size
        self.valid_size = valid_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.episode_length = episode_length
        self.buffer = ReplayBuffer(replay_buffer_size)
        self.w1 = w1
        self.w2 = w2
        update_fn = None
        update_mask = None

        self.episode_rewards = 0
        self.total_reward = 0

        self.actor = Actor(static_size,
                           dynamic_size,
                           self.hidden_size,
                           update_fn,
                           update_mask,
                           self.num_layers,
                           self.dropout).to(device)

        self.critic = Critic(static_size,
                             dynamic_size,
                             self.hidden_size).to(device)

        self.agent = Agent()

    def model_loss(self,reward,critic_est,tour_logp):
        """Function to calculate the losses for the actor and critic.

        Args:
            reward (float): The actual reward for the RSU network.
            critic_est (float): The estimate of the reward for the RSU network from the critic.
            tour_logp (float): Log probabilities for each of the selected RSUs in the network.

        Returns:
            _type_: _description_
        """
        advantage = (reward-critic_est)
        actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
        critic_loss = torch.mean(advantage ** 2)
        return actor_loss, critic_loss

    def forward(self,x: Tensor) -> Tensor:
        rsu_indicies, _ = self.actor(x)
        return rsu_indicies

    def training_step(self,batch:Tuple[Tensor,Tensor],batch_idx,optimizer_idx) -> OrderedDict:
        static, dynamic, x0 = batch

        rsu_indices, rsu_logp = self.actor(static, dynamic, x0)

        reward, done = self.agent.simulation_step(rsu_indices)
        self.episode_rewards += reward

        critic_est = self.critic(static,dynamic).view(-1)

        actor_loss, critic_loss = self.model_loss(reward,critic_est,rsu_logp)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        if optimizer_idx == 0: loss = actor_loss
        if optimizer_idx == 1: loss = critic_loss

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})
        
    def configure_optimizers(self) -> List[Optimizer]:
        """Initializes Adam optimizers for actor and critic.

        Returns:
            List[Optimizer]: Optimizers for actor and critic.
        """
        actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)
        return [actor_optim, critic_optim]

    def train_dataloader(self) -> DataLoader:
        """Training loader for experience data, used for experience replay training.

        Returns:
            DataLoader: Loader for experience dataset.
        """
        return self.__dataloader()

    def __dataloader(self) -> DataLoader:
        """Initializes the replay buffer used for retrieving experiences.

        Returns:
            DataLoader: Loader for experience dataset.
        """
        dataset = self.RLDDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(dataset=dataset,batch_size=self.batch_size)
        return dataloader