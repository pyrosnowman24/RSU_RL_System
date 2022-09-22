from ctypes import pointer
from json import encoder
from tkinter import Variable
from tkinter.filedialog import askdirectory
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
import signal
import sys
import os
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent
from Models.Reward_Modeling.Reward_Modeling_Transformer import Reward_Modeling
from Models.RSU_Intersections_Datamodule import RSU_Intersection_Datamodule


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

class RW_System(LightningModule):
    def __init__(self,
                 agent,
                 num_features: int = 4,
                 nhead: int = 4,
                 batch_size: int = 100,
                 max_num_rsu: int = 20,
                 episodes_per_epoch: int = 200,
                 W: int = [.5,.5],
                 n_layers: int = 1,
                 lr = 1e-4,
                 model_directory = "/home/acelab/",
                 save_data_bool:bool = False):

        super(RW_System,self).__init__()
        
        self.num_features = num_features
        self.nhead = nhead
        self.lr = lr

        self.max_num_rsu = max_num_rsu
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.W = W
        self.n_layers = n_layers

        self.agent = agent

        self.reward_modeling = Reward_Modeling(self.num_features,
                                         self.nhead,
                                         n_layers = self.n_layers)

        # self.df_history = pd.DataFrame(columns=["intersections","intersection_idx","pre_rsu_network","rsu_network","reward","loss"],dtype=object)
        self.df_history = pd.DataFrame(columns=["intersection_idx","rsu_network","reward","critic_reward","loss"],dtype=object)
        self.df_new_data = pd.DataFrame(columns=["intersection_idx","rsu_network","reward","critic_reward","loss"],dtype=object)
        self.model_directory = model_directory
        self.model_history_file = os.path.join(self.model_directory,"model_history.csv")
        self.save_data_bool = save_data_bool

        self.running_average = 0

    def forward(self,x):
        return self.reward_modeling(x)

    def training_step(self,batch):
        intersections = batch[0][:,:,1:]
        intersection_idx = batch[1]
        rsu_network_idx = batch[2]
        # rsu_network_idx[rsu_network_idx>0] -= 1
        mask = batch[3]
        q_values, pointer_argmaxs, new_mask = self.reward_modeling(intersections,intersection_idx,rsu_network_idx,mask.clone())
        rsu_idx = pointer_argmaxs[pointer_argmaxs>0] # the -1 shifts the indices to the left, which is needed when the first option is removed

        rsu_network = intersections[:,rsu_idx]


        rewards = self.agent.simulation_step(rsu_idx-1,intersection_idx[0,1:],model = "Q Learning Positive")
        rewards = torch.tensor([np.array(rewards)],requires_grad=True,dtype=torch.float)
        
        rewards_critic = self.reward_modeling.reward_network(rsu_network)

        print("rewards",rewards)
        print("critics rewards",rewards_critic)
        
        loss = self.loss(rewards, rewards_critic)

        print("loss",loss)

        # data = [intersections,intersection_idx,rsu_network_idx,rsu_idx,reward.detach().numpy(),loss.detach().numpy()]
        intersection_test = intersection_idx.detach().numpy()[0,:]
        rsu_idx_test = pointer_argmaxs.detach().numpy()
        rsu_idx_test = rsu_idx_test[rsu_idx_test>0]
        reward_test = rewards.detach().numpy()
        critic_reward_test = rewards_critic.detach().numpy()
        loss_test = loss.detach().numpy()

        data = np.array((intersection_test,rsu_idx_test,reward_test,critic_reward_test,loss_test),dtype=object)
        self.df_history.loc[self.df_history.shape[0]] = data
        self.df_new_data.loc[0] = data
        if self.save_data_bool:
            self.save_data()
        return loss.float()

    # def training_epoch_end(self,output):
    #     self.reward_modeling.reward_network.load_state_dict(self.reward_modeling.network.state_dict())

    def configure_optimizers(self):
        policy_gradient_optim = Adam(self.reward_modeling.reward_network.parameters(),lr = self.lr)
        return policy_gradient_optim

    def pad_reward(self,reward,pointer_argmax):
        reward_padded = np.zeros(shape = pointer_argmax.shape)
        for i, index in enumerate(pointer_argmax[pointer_argmax>0]):
            reward_padded[0,index] = reward[0,i]
        return torch.tensor(reward_padded,dtype=torch.float)

    def loss(self,rewards, rewards_critic, gamma: float = 0.99):
        """Calculates the loss for the Q-Learning RL algorithm.

        Args:
            batch (list): The batch of intersections, intersection idx, rsu idx, and mask

        Returns:
            loss (torch.Tensor): The loss of the solution for the Policy Gradient RL algorithm. The higher the loss, the worse the solution. 
        """

        return nn.MSELoss()(rewards,rewards_critic)

    def save_data(self):
        print("Saving Data")
        if not os.path.isfile(self.model_history_file):
            self.df_new_data.to_csv(self.model_history_file, index=False)
        else: # else it exists so append without writing the header
            self.df_new_data.to_csv(self.model_history_file, index=False, mode='a', header=False)

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
    model_name = "Training_for_Reward_4features"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_name = "Training_for_Reward"
    checkpoint_directory = os.path.join(directory_path,checkpoint_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,checkpoint_name)


    if save_model_bool:
            os.makedirs(model_directory)
    if train_new_model:
        simulation_agent = Agent()
        model = RW_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool)
        trainer = Trainer(max_epochs = max_epochs)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)

    else:
        model = RW_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool)
        model.load_state_dict(torch.load(checkpoint_path))
        trainer = Trainer(max_epochs = max_epochs)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)
        
