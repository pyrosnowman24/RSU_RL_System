from json import encoder
from tkinter import Variable
from typing import List, Tuple, Callable, Iterator
from collections import OrderedDict, deque, namedtuple
import sys
import os

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import IterableDataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent
from Models.Actor_Critic.Actor_Critic import Actor_Critic
from Models.RSU_Intersections_Datamodule import RSU_Intersection_Datamodule

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
    def __init__(self, 
                     agent,
                     num_features: int = 4,
                     num_layers: int = 1, 
                     actor_lr: float = 5e-4,
                     critic_lr: float = 5e-4, 
                     W: int = [.5,.5]):

        super(DRL_System,self).__init__()
        self.num_layers = num_layers
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.W = W

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

    def forward(self,intersections: Tensor, mask: Tensor) -> Tensor:
        _, _, rsu_network, critic_reward = self.actor_critic(intersections, mask)
        return rsu_network, critic_reward

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

def save_model(model,model_directory,model_path):
        model_history_file = os.path.join(model_directory,"model_history.csv")
        torch.save(model.state_dict(),model_path)
        
if __name__ == '__main__':
    max_epochs = 25
    train_new_model = True
    save_model_bool = False
    display_figures = True

    simulation_agent = Agent()
    trainer = Trainer(max_epochs = max_epochs)
    directory_path = "/home/acelab/Dissertation/RSU_RL_Placement/trained_models/"
    model_name = "Training_for_Reward"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_name = "Training_for_Reward"
    checkpoint_directory = os.path.join(directory_path,checkpoint_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,checkpoint_name)

    if save_model_bool:
            os.makedirs(model_directory)
    if train_new_model:
        simulation_agent = Agent()
        model = DRL_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool)
        trainer = Trainer(max_epochs = max_epochs)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)

    else:
        model = DRL_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool)
        model.load_state_dict(torch.load(checkpoint_path))
        trainer = Trainer(max_epochs = max_epochs)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)
