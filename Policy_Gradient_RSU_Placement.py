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

from Agent import Agent
from Policy_Gradient_Transformer import Policy_Gradient
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

class PG_System(LightningModule):
    def __init__(self,
                 agent,
                 num_features: int = 3,
                 nhead: int = 4,
                 batch_size: int = 100,
                 max_num_rsu: int = 20,
                 episodes_per_epoch: int = 200,
                 W: int = [.5,.5],
                 n_layers: int = 1,
                 lr = 1e-4):

        super(PG_System,self).__init__()
        
        self.num_features = num_features
        self.nhead = nhead
        self.lr = lr

        self.max_num_rsu = max_num_rsu
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.W = W
        self.n_layers = n_layers

        self.agent = agent

        self.policy_gradient = Policy_Gradient(self.num_features,
                                         self.nhead,
                                         n_layers = self.n_layers)

        self.loss_history = []
        self.rsu_network_history = []


    def forward(self,x):
        return self.policy_gradient(x)

    def training_step(self,batch):
        intersections = batch[0]
        intersection_idx = batch[1]
        rsu_network_idx = batch[2]
        mask = batch[3]
        log_pointer_scores, pointer_argmaxs = self.policy_gradient(intersections,intersection_idx,rsu_network_idx,mask)
        rsu_idx = pointer_argmaxs[pointer_argmaxs>0]-1
        self.rsu_network_history.append(pointer_argmaxs)
        print(rsu_idx)
        print(intersection_idx[0,1:])
        if len(rsu_idx) != 0:
            reward = simulation_agent.simulation_step(rsu_idx,intersection_idx[0,1:],model = "Policy Gradient")
        else:
            reward = torch.tensor(0.0)
        reward = torch.tensor([reward],requires_grad=True,dtype=torch.float)
        loss = self.loss(log_pointer_scores,pointer_argmaxs,reward)
        print("loss",loss)
        self.loss_history.append(loss.detach().numpy())
        return loss
    
    def validation_step(self,batch):
        print(3)

    def configure_optimizers(self):
        policy_gradient_optim = Adam(self.policy_gradient.parameters(),lr = self.lr)
        return policy_gradient_optim

    def loss(self,log_prob,rsu_idx,rewards):
        if torch.sum(rsu_idx) == 0.0:
            return torch.tensor(-10.0,requires_grad=True)
        padded_rewards = torch.zeros(rsu_idx.shape[1])
        padded_rewards[rsu_idx[0,:] != 0] = rewards
        log_prob_actions = padded_rewards * log_prob[0,range(log_prob.shape[1]),rsu_idx]
        log_prob_actions = log_prob_actions[log_prob_actions!=0.0]
        loss = -log_prob_actions.mean()
        return loss

def save_information(model,loss_history,rsu_network_history):
    model_directory = os.path.join(directory_path,model_name,'/')
    model_path = os.path.join(model_directory,model_name)
    loss_history_file = os.path.join(model_directory,"loss_history.csv")
    rsu_network_history_file = os.path.join(model_directory,"rsu_network_history.csv")

    torch.save(model.state_dict(),model_path)
    os.makedirs(model_directory)
    np.savetxt(loss_history_file, loss_history, delimiter=",")
    np.savetxt(rsu_network_history_file, rsu_network_history, delimiter=",")


if __name__ == '__main__':
    max_epochs = 2
    train_new_model = True
    save_model = True
    save_data = True
    display_figures = True
    directory_path = "/home/acelab/Dissertation/RSU_RL_Placement/trained_models/"
    model_name = "test"
    model_directory = os.path.join(directory_path,model_name,'/')
    model_path = os.path.join(model_directory,model_name)

    simulation_agent = Agent()
    trainer = Trainer(max_epochs = max_epochs)
    if train_new_model:
        simulation_agent = Agent()
        model = PG_System(simulation_agent)
        trainer = Trainer(max_epochs = 1)
        datamodule = RSU_Intersection_Datamodule(simulation_agent,batch_size=1)
        trainer.fit(model,datamodule=datamodule)
        if save_model:
            torch.save(model.state_dict())


    else:
        model = PG_System(simulation_agent)
        model.load_state_dict(torch.load(model_path))
        model.eval()