from cmath import isnan, nan
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
from Models.Policy_Gradient.Policy_Gradient_Transformer import Policy_Gradient
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

class PG_System(LightningModule):
    def __init__(self,
                 agent,
                 num_features: int = 3,
                 nhead: int = 4,
                 W: int = [.5,.5],
                 n_layers: int = 1,
                 lr = 1e-4,
                 model_directory = "/home/acelab/",
                 save_data_bool:bool = False):

        super(PG_System,self).__init__()
        
        self.num_features = num_features
        self.nhead = nhead
        self.lr = lr
        self.W = W
        self.n_layers = n_layers

        self.agent = agent

        self.policy_gradient = Policy_Gradient(self.num_features,
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
        return self.policy_gradient(x)

    def training_step(self,batch):
        intersections = batch[0]
        intersection_idx = batch[1]
        rsu_network_idx = batch[2]
        mask = batch[3]
        print("intersection idx",intersection_idx)
        print("og rsu idx",rsu_network_idx)
        log_pointer_scores, pointer_argmaxs = self.policy_gradient(intersections,intersection_idx,rsu_network_idx,mask)
        rsu_idx = pointer_argmaxs[pointer_argmaxs>0]-1
        print("rsu_idx ",rsu_idx)
        if len(rsu_idx) != 0:
            print("With RSU network")
            reward = self.agent.simulation_step(rsu_idx,intersection_idx[0,1:],model = "Policy Gradient")
            reward = torch.tensor([reward],requires_grad=True,dtype=torch.float)
            # self.running_average = (reward.detach().numpy().mean() + self.running_average)/2
            # scaled_reward = reward - self.running_average
            loss = self.loss(log_pointer_scores,pointer_argmaxs,reward)
        else:
            print("Without RSU network")
            reward = torch.tensor([0.0],requires_grad=True,dtype=torch.float)
            # self.running_average = (reward.detach().numpy().mean() + self.running_average)/2
            # scaled_reward = reward - self.running_average
            loss = self.loss(log_pointer_scores,pointer_argmaxs,reward)
        print("reward",reward)
        print("running average",self.running_average)
        # data = [intersections,intersection_idx,rsu_network_idx,rsu_idx,reward.detach().numpy(),loss.detach().numpy()]
        print("rsu idx",rsu_idx)
        print("loss",loss.detach().numpy())
        data = [intersection_idx.numpy(),rsu_idx.numpy(),reward.detach().numpy(),loss.detach().numpy()]
        print("data",data)
        if self.df_new_data.shape[0] != 0: print(self.df_new_data.loc[0])
        self.df_history.loc[self.df_history.shape[0]] = data
        self.df_new_data.loc[0] = data
        print(self.df_new_data.loc[0])
        if self.save_data_bool:
            self.save_data()
        return loss

    def configure_optimizers(self):
        policy_gradient_optim = Adam(self.policy_gradient.parameters(),lr = self.lr)
        return policy_gradient_optim

    def loss(self,log_prob,rsu_idx,rewards):
        """Calculates the loss for the Policy Gradient RL algorithm. The loss only takes into account the RSUs selected by 
        the algorithm, not the RSUs that were already in the environment. This is because the log probability of 1 
        (the RSUs are guarenteed to be present) is 0, so they wouldnt have an effect.

        Args:
            log_prob (torch.Tensor): The log probability of selecting each intersection for each step.
            rsu_idx (torch.Tensor): The index of the selected RSU locations from the intersections.
            rewards (torch.Tensor): The reward for each RSU in the network.

        Returns:
            loss (torch.Tensor): The loss of the solution for the Policy Gradient RL algorithm. The higher the loss, the worse the solution. 
        """
        padded_rewards = torch.zeros(rsu_idx.shape[1])
        padded_rewards[rsu_idx[0,:] != 0] = rewards
        # print(-log_prob[0,range(log_prob.shape[1]),rsu_idx])
        # print(padded_rewards)
        log_prob_actions = padded_rewards * -log_prob[0,range(log_prob.shape[1]),rsu_idx]
        log_prob_actions = torch.nan_to_num(log_prob_actions,0.0)
        log_prob_actions = log_prob_actions[log_prob_actions!=0.0]
        # print(log_prob_actions)
        loss = log_prob_actions.nanmean()
        # print(loss)

        if np.isnan(loss.detach().numpy()):
            loss = torch.tensor(10000.0,requires_grad=True)
        return loss

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
    max_epochs = 1
    train_new_model = True
    save_model_bool = True
    display_figures = True
    simulation_agent = Agent()
    trainer = Trainer(max_epochs = max_epochs)
    directory_path = "/home/acelab/Dissertation/RSU_RL_Placement/trained_models/"
    model_name = "test"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_name = "Training_for_Reward"
    checkpoint_directory = os.path.join(directory_path,checkpoint_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,checkpoint_name)


    if save_model_bool:
            os.makedirs(model_directory)
    if train_new_model:
        simulation_agent = Agent()
        model = PG_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool)
        trainer = Trainer(max_epochs = max_epochs)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)

    else:
        model = PG_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool)
        model.load_state_dict(torch.load(checkpoint_path))
        trainer = Trainer(max_epochs = max_epochs)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)
