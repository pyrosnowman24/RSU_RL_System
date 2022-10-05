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
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Models.Agent import Agent
from Models.Q_Learning.Q_Learning_Transformer import Q_Learning
from Models.RSU_Intersections_Datamodule import RSU_Intersection_Datamodule

directory_path = os.getcwd()
model_name = "rsu_performance_dataset"
model_directory = os.path.join(directory_path,model_name)


def create_new_performance_model():
    simulation_agent = Agent()

    num_intersections = simulation_agent.network_intersections.shape[0]

    df_data = pd.DataFrame(columns=["intersection_id","x","y","z","size","feature","reward"],dtype=object)
    step = 100

    for current in np.arange(num_intersections,step = step):
        if current+step < num_intersections:
            next = current+step
        else: next = num_intersections
        intersection_idx = np.arange(current, next)
        rsu_idx = np.arange(intersection_idx.shape[0])
        reward, features = simulation_agent.simulation_step(rsu_idx,intersection_idx,model = "Q Learning Positive")
        for i in np.arange(intersection_idx.shape[0]):
            intersection_data = simulation_agent.network_intersections[intersection_idx[i]]
            data = np.array((intersection_data[0].detach().numpy(),
                            intersection_data[1].detach().numpy(),
                            intersection_data[2].detach().numpy(),
                            intersection_data[3].detach().numpy(),
                            intersection_data[4].detach().numpy(),
                            features[:,i],
                            reward[i]),
                            dtype=object)
            df_data.loc[len(df_data.index)] = data

    print(model_directory)
    df_data.to_csv(model_directory, index=False)

def plot_performance_model():
    df = pd.read_csv(model_directory)
    rewards = df['reward'].to_numpy()
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    # fig,ax = plt.subplots(1)
    # ax.scatter(np.arange(rewards_filtered.shape[0]),rewards_filtered)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x,y,rewards)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("reward")
    plt.show()


create_new_performance_model()