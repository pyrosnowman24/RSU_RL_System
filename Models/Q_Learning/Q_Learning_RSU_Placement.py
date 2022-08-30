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

from Agent import Agent
from Models.Q_Learning.Q_Learning_Transformer import Q_Learning
from RSU_Intersections_Datamodule import RSU_Intersection_Datamodule


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

class QL_System(LightningModule):
    def __init__(self,
                 agent,
                 num_features: int = 3,
                 nhead: int = 4,
                 batch_size: int = 100,
                 max_num_rsu: int = 20,
                 episodes_per_epoch: int = 200,
                 W: int = [.5,.5],
                 n_layers: int = 1,
                 lr = 1e-4,
                 model_directory = "/home/acelab/",
                 save_data_bool:bool = False):

        super(QL_System,self).__init__()
        
        self.num_features = num_features
        self.nhead = nhead
        self.lr = lr

        self.max_num_rsu = max_num_rsu
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.W = W
        self.n_layers = n_layers

        self.agent = agent

        self.q_learning = Q_Learning(self.num_features,
                                         self.nhead,
                                         n_layers = self.n_layers)

        # self.df_history = pd.DataFrame(columns=["intersections","intersection_idx","pre_rsu_network","rsu_network","reward","loss"],dtype=object)
        self.df_history = pd.DataFrame(columns=["intersection_idx","rsu_network","reward","loss"],dtype=object)
        self.df_new_data = pd.DataFrame(columns=["intersection_idx","rsu_network","reward","loss"],dtype=object)
        self.model_directory = model_directory
        self.model_history_file = os.path.join(self.model_directory,"model_history.csv")
        self.save_data_bool = save_data_bool

        self.running_average = 0

    def forward(self,x):
        return self.q_learning(x)

    def training_step(self,batch):
        intersections = batch[0]
        intersection_idx = batch[1]
        rsu_network_idx = batch[2]
        mask = batch[3]
        q_values, pointer_argmaxs = self.q_learning(intersections,intersection_idx,rsu_network_idx,mask)
        rsu_idx = pointer_argmaxs[pointer_argmaxs>0]-1
        # print(rsu_network_idx)
        # print(rsu_idx)
        # print(intersection_idx[0,:])
        if len(rsu_idx) != 0:
            reward = self.agent.simulation_step(rsu_idx,intersection_idx[0,1:],model = "Policy Gradient")
            reward = torch.tensor([reward],requires_grad=True,dtype=torch.float)
            self.running_average = (reward.detach().numpy().mean() + self.running_average)/2
            scaled_reward = reward - self.running_average
            loss = self.loss(intersections,q_values,scaled_reward)
        else:
            reward = torch.tensor([0.0],requires_grad=True,dtype=torch.float)
            self.running_average = (reward.detach().numpy().mean() + self.running_average)/2
            scaled_reward = reward - self.running_average
            loss = self.loss(intersections,q_values,scaled_reward)

        # data = [intersections,intersection_idx,rsu_network_idx,rsu_idx,reward.detach().numpy(),loss.detach().numpy()]
        data = [intersection_idx,rsu_idx,scaled_reward.detach().numpy(),loss.detach().numpy()]
        self.df_history.loc[self.df_history.shape[0]] = data
        self.df_new_data.loc[0] = data
        if self.save_data_bool:
            self.save_data()
        return loss

    def configure_optimizers(self):
        policy_gradient_optim = Adam(self.q_learning.parameters(),lr = self.lr)
        return policy_gradient_optim

    def loss(self,intersections,q_values,rewards, gamma: float = 0.99):
        """Calculates the loss for the Q-Learning RL algorithm.

        Args:
            batch (list): The batch of intersections, intersection idx, rsu idx, and mask

        Returns:
            loss (torch.Tensor): The loss of the solution for the Policy Gradient RL algorithm. The higher the loss, the worse the solution. 
        """

        with torch.no_grad():
            next_state_values = self.q_learning.target_network(intersections)
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards
        return nn.MSELoss()(q_values,expected_state_action_values)

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
    save_model_bool = True
    display_figures = True
    simulation_agent = Agent()
    trainer = Trainer(max_epochs = max_epochs)
    directory_path = "/home/acelab/Dissertation/RSU_RL_Placement/trained_models/"
    model_name = "25_Epochs_30_Intersections_Scaled"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)
    if save_model_bool:
            os.makedirs(model_directory)
    if train_new_model:
        simulation_agent = Agent()
        model = QL_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool)
        trainer = Trainer(max_epochs = max_epochs)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)

    else:
        model = QL_System(simulation_agent)
        model.load_state_dict(torch.load(model_path))
        model.eval()
