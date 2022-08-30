from json import encoder
from tkinter import Variable
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

from Agent import Agent
from Actor_Critic import Actor_Critic, Actor, Critic

import numpy as np
import argparse

# This is a advantage actor-critic model


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
                       batch_size: int = 100,
                       num_layers: int = 1, 
                       dropout: float = 0.1, 
                       actor_lr: float = 5e-4,
                       critic_lr: float = 5e-4, 
                       state_size: int = 3,
                       max_num_rsu: int = 20,
                       episodes_per_epoch: int = 200,
                       number_nodes: int = 100,
                       W: int = [.5,.5]):
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
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.state_size = state_size
        self.max_num_rsu = max_num_rsu
        self.episodes_per_epoch = episodes_per_epoch
        self.W = W

        self.agent = Agent()

        self.actor = Actor(state_size,
                           self.hidden_size,
                           self.device,
                           self.num_layers,
                           self.dropout).to(self.device)

        self.critic = Critic(state_size,
                             self.hidden_size).to(self.device)

        

        self.actor_critic = Actor_Critic(self.actor, self.critic, self.agent, self.device)

        self.episode_states = np.empty((0,self.agent.state.shape[1]))
        self.episode_actions = np.empty((0,self.agent.state.shape[1]))
        self.episode_logp = np.empty((0,1))
        self.episode_adv = []
        self.episode_rewards = []
        self.episode_critic_rewards = []

        self.batch_states = []
        self.batch_actions = []
        self.batch_logp = []
        self.batch_adv = []

        self.epoch_rewards = []

        self.avg_rewards = 0
        self.avg_ep_rewards = 0

        _ = self.agent.reset()

    def calculate_advantage(self,rewards,critic_ests):
        """Function to calculate the losses for the actor and critic.

        Args:
            rewards (float): The actual rewards for the RSU network.
            critic_est (float): The estimate of the rewards for the RSU network from the critic.

        Returns:
            _type_: _description_
        """
        advantage = [(rewards[i] - critic_ests[i])[0][0] for i in range(len(rewards))]
        return advantage

    def actor_loss(self,advantage,logps) -> torch.Tensor:
        actor_loss = torch.mean(torch.mul(advantage.detach(),torch.sum(logps)))
        actor_loss.requires_grad = True
        return actor_loss

    def critic_loss(self,advantage) -> torch.Tensor:
        critic_loss = torch.mean(torch.pow(advantage.detach(),2))
        critic_loss.requires_grad = True
        return critic_loss

    def forward(self,x: Tensor) -> Tensor:
        # This must be redone based on the new actor call method, it should still return an RSU network
        rsu_indicies, _, _ = self.actor_critic(x)
        return rsu_indicies

    def training_step(self,batch:Tuple[Tensor,Tensor],batch_idx,optimizer_idx) -> OrderedDict:
        states, actions, logps, advantages = batch
        # advantages = (advantages - advantages.mean())/advantages.std()
        print(states.shape,'\n', actions.shape,'\n', logps.shape,'\n', advantages.shape,'\n') # Shape is (Batch_size,size of RSU network, number of intersections)
        self.log("avg_ep_reward", self.avg_ep_rewards, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_reward", self.avg_rewards, prog_bar=True, on_step=False, on_epoch=True)

        if optimizer_idx == 0: 
            loss_actor = self.actor_loss(advantages,logps)
            self.log('loss_actor', loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # print('\n',"Loss Actor",loss_actor)
            return loss_actor

        if optimizer_idx == 1: 
            loss_critic = self.critic_loss(advantages)
            self.log('loss_critic', loss_critic, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # print("Loss critic",loss_critic,'\n')
            return loss_critic
     
    def configure_optimizers(self) -> List[Optimizer]:
        """Initializes Adam optimizers for actor and critic.

        Returns:
            List[Optimizer]: Optimizers for actor and critic.
        """
        actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)
        return actor_optim, critic_optim

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
        for episode in range(self.episodes_per_epoch):
            terminate_simulation = False # Tests if the episode should end
            epoch_end = False # Tests if the epoch should end
            while not terminate_simulation:
                self.episode_states = np.vstack((self.episode_states,self.agent.state.numpy()[0]))
                action, logp, critic_reward = self.actor_critic(self.agent.state[None,:,:]) 
                
                quit()
                _, reward, simulation_done = self.agent.simulation_step(action,self.W)
                
                self.episode_logp = np.vstack((self.episode_logp,logp.detach().numpy()[0][0]))

                self.episode_actions = np.vstack((self.episode_actions, action.detach().numpy()))

                self.episode_rewards.append(reward)
                self.episode_critic_rewards.append(critic_reward.detach().numpy())

                # Tests if all the samples for the batch are generated
                # Tests if the max number of RSUs have been selected or all intersections have been selected
                terminate_simulation = (self.agent.state.shape[1] - torch.sum(self.agent.state)) == self.max_num_rsu or torch.sum(self.agent.state) == 0.0
                epoch_end = episode == self.episodes_per_epoch-1

                if simulation_done or terminate_simulation:

                    _ = self.agent.reset()
                    self.actor_critic.reset()

                    self.epoch_rewards.append(np.sum(self.episode_rewards))
                    self.episode_adv = np.array(self.calculate_advantage(self.episode_rewards,self.episode_critic_rewards),ndmin=2).transpose()

                    self.batch_states.append(torch.from_numpy(self.episode_states))
                    self.batch_actions.append(torch.from_numpy(self.episode_actions))
                    self.batch_logp.append(torch.from_numpy(self.episode_logp))
                    self.batch_adv.append(torch.from_numpy(self.episode_adv))
                        
                    self.episode_states = np.empty((0,self.agent.state.shape[1]))
                    self.episode_actions = np.empty((0,self.agent.state.shape[1]))
                    self.episode_logp = np.empty((0,1))
                    self.episode_adv = []
                    self.episode_rewards = []
                    self.episode_critic_rewards = []
            if epoch_end:
                # Yield training data for model
                # print(self.batch_states,'\n', self.batch_actions,'\n',self.batch_logp,'\n',self.batch_adv,'\n')
                train_data = zip(self.batch_states, self.batch_actions,self.batch_logp,self.batch_adv)
                for state, action, logp, advantages in train_data:
                    print(state,'\n', action,'\n', logp,'\n', advantages,'\n')
                    yield state, action, logp, advantages

                # Reset history for next epoch
                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_logp.clear()
                self.batch_adv.clear()

            # Logging
            self.avg_rewards = sum(self.epoch_rewards) / self.episodes_per_epoch

            total_epoch_reward = np.sum(self.epoch_rewards)
            nb_episodes = len(self.epoch_rewards)
            self.avg_ep_rewards = total_epoch_reward / nb_episodes

            self.epoch_rewards = []

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--number_nodes', default=50, type=float)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=120000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)
    parser.add_argument('--max_num_rsu', default=20, type=int)
    parser.add_argument('--episodes_per_epoch', default=32, type=int)
    args, unknown = parser.parse_known_args()
    T = 100
    w2_list = np.arange(T+1)/T
    w1_list = 1-w2_list

    model = DRL_System(**args.__dict__)
    trainer = Trainer(max_epochs = 1)
    trainer.fit(model)
