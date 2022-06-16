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
        print(2)

    def validation_step(self,batch):
        print(3)

    def configure_optimizers(self):
        actor_critic_optim = Adam(self.actor_critic.parameters(),lr = self.lr)
        return actor_critic_optim


simulation_agent = Agent()
model = DRL_System(simulation_agent)
trainer = Trainer(max_epochs = 50)
datamodule = test_datamodule = RSU_Intersection_Datamodule(simulation_agent,batch_size=3)

trainer.fit(model,datamodule=datamodule)