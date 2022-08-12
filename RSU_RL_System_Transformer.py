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

from Agent import Agent
from Actor_Critic_Transformer import Actor_Critic
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

class DRL_System(LightningModule):
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

        super(DRL_System,self).__init__()
        
        self.num_features = num_features
        self.nhead = nhead
        self.lr = lr

        self.max_num_rsu = max_num_rsu
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.W = W
        self.n_layers = n_layers

        self.agent = agent

        self.actor_critic = Actor_Critic(self.num_features,
                                         self.nhead,
                                         n_layers = self.n_layers)

    def forward(self,x):
        return self.actor_critic(x)

    def training_step(self,batch):
        intersections = batch[0]
        intersection_idx = batch[1]
        rsu_network_idx = batch[2]
        mask = batch[3]
        log_pointer_scores, pointer_argmaxs = self.actor_critic(intersections,intersection_idx,rsu_network_idx,mask)
        rsu_idx = pointer_argmaxs[pointer_argmaxs>0]-1
        print(rsu_idx)
        print(intersection_idx[0,1:])
        if len(rsu_idx) != 0:
            reward = simulation_agent.simulation_step(rsu_idx,intersection_idx[0,1:])
        else:
            reward = torch.tensor(0.0)
        reward = torch.tensor([reward])
        # return reward

    def validation_step(self,batch):
        print(3)

    def configure_optimizers(self):
        actor_critic_optim = Adam(self.actor_critic.parameters(),lr = self.lr)
        return actor_critic_optim

# def exit_gracefully(signum,frame):
#     signal.signal(signal.SIGINT, original_sigint)
#     try:
#         simulation_agent.kill_sumo_env()
#         sys.exit(1)

#     except KeyboardInterrupt:
#         print("Ok ok, quitting")
#         sys.exit(1)

if __name__ == '__main__':
    # original_sigint = signal.getsignal(signal.SIGINT)
    # signal.signal(signal.SIGINT, exit_gracefully)
    simulation_agent = Agent()
    model = DRL_System(simulation_agent)
    trainer = Trainer(max_epochs = 50)
    datamodule = RSU_Intersection_Datamodule(simulation_agent,batch_size=1)

    trainer.fit(model,datamodule=datamodule)