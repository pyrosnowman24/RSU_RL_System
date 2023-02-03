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
import re
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import stats, special
from sklearn.preprocessing import MinMaxScaler

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Models.reward import RewardFunctions
from Models.Agent import Agent
from Models.Q_Learning.Q_Learning_Transformer import Q_Learning
from Models.RSU_Intersections_Datamodule import RSU_Intersection_Datamodule

directory_path = os.getcwd()
model_name = "intersection_performance_dataset_new"
model_directory = os.path.join(directory_path,model_name)
reward_functions = RewardFunctions()
# df = pd.read_csv(model_directory)


def create_new_performance_model():
    simulation_agent = Agent()

    num_intersections = simulation_agent.network_intersections.shape[0]

    df_data = pd.DataFrame(columns=["intersection_id","x","y","z","degree_centrality","closeness_centrality","avg_dbm","num_contacts","reward1","reward2","reward3","reward4","reward_sum"],dtype=object)
    step = 100

    for current in np.arange(num_intersections,step = step):
        if current+step < num_intersections:
            next = current+step
        else: next = num_intersections
        intersection_idx = np.arange(current, next)
        rsu_idx = np.arange(intersection_idx.shape[0])
        rewards, reward, features = simulation_agent.simulation_step(rsu_idx,intersection_idx,model = "Q Learning Positive")
        for i in np.arange(intersection_idx.shape[0]):
            intersection_data = simulation_agent.network_intersections[intersection_idx[i]]
            data = np.array((intersection_data[0].detach().numpy(),
                            intersection_data[1].detach().numpy(),
                            intersection_data[2].detach().numpy(),
                            intersection_data[3].detach().numpy(),
                            intersection_data[4].detach().numpy(),
                            intersection_data[5].detach().numpy(),
                            features[0,i],
                            features[1,i],
                            rewards[0,i],
                            rewards[1,i],
                            rewards[2,i],
                            rewards[3,i],
                            reward[i]),
                            dtype=object)
            print(data)
            df_data.loc[len(df_data.index)] = data
    df_data.to_csv(model_directory, index=False)

def plot_performance_model():
    rewards = df['reward_sum'].to_numpy()
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    size = df['size'].to_numpy()
    fig,ax = plt.subplots(1)
    ax.scatter(df['rewards3'].to_numpy(),rewards)
    plt.show()

def plot_rewards_xy():
    rewards = df['reward'].to_numpy()
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    size = df['size'].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x,y,rewards)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("reward")
    plt.show()

    fig,ax=plt.subplots(1)
    ax.scatter(size,rewards)
    ax.set_xlabel("# Roads Connected to Intersection")
    ax.set_ylabel("Reward")
    ax.set_title("# Roads Connected to Intersection vs. Reward")
    plt.show()

def plot_reward_skews():
    lmbda = 0.24098677879102673 # Variable determined by algorithm originally based on sampled rewards from entire data.

    rewards = df['reward'][df['reward'] > .01]

    print("Original Skewness",df['reward'].skew())
    print("Original Shapiro",stats.shapiro(df['reward'])[1])
    print('\n')

    reward_inv = 1/(rewards)
    print("Inv Transform",reward_inv.skew())
    print("Inv Shapiro",stats.shapiro(reward_inv)[1])
    print('\n')

    reward_log = np.log(rewards)
    print("Log Transform",reward_log.skew())
    print("Log Shapiro",stats.shapiro(reward_log)[1])
    print('\n')

    reward_sqrt = np.sqrt(rewards)
    print("SQRT Transform",reward_sqrt.skew())
    print("SQRT Shapiro",stats.shapiro(reward_sqrt)[1])
    print('\n')

    reward_boxcox,lmbda = stats.boxcox(rewards)
    reward_boxcox = pd.Series(reward_boxcox)
    print("Box Cox Transform",reward_boxcox.skew())
    print("Box Cox Shapiro",stats.shapiro(reward_boxcox)[1])
    print('\n')

    inv_boxcox = special.inv_boxcox(reward_boxcox,lmbda)
    print("Inverse Box Cox Transform",inv_boxcox.skew())
    
    print(df['reward'].shape)
    print(reward_sqrt.shape)
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.hist(reward_inv)
    ax2.hist(reward_log)
    ax3.hist(reward_sqrt)
    ax4.hist(reward_boxcox)
    plt.show()
    
def plot_xy_coords():
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    scaler = MinMaxScaler()
    xy_scaled = scaler.fit_transform(df[['x','y']])
    x_scaled = xy_scaled[:,0]
    y_scaled = xy_scaled[:,1]

    fig,(ax1,ax2) = plt.subplots(2)
    ax1.scatter(x,y)
    ax2.scatter(x_scaled,y_scaled)
    plt.show()

def rewards_preprocessing():
    features = df['feature']
    reward = np.empty(shape = features.shape,dtype=object)
    for i in range(reward.shape[0]):
        print(features[i])
        results = re.findall(r'[-+]?\d*\.?\d+',features[i])
        reward[i] = [float(x) for x in results]
    print(reward)
    
    # reward_sqrt = np.sqrt(rewards)
    # print(reward_sqrt.skew())
    # print(stats.shapiro(reward_sqrt)[1])

    # fig,ax = plt.subplots(1)
    # ax.hist(reward_sqrt)
    # plt.show()

def reward_positive_ql(self,features):
    """Reward for Q-Learning based model. The higher the reward the better the solution. This version will always be positive.

    Args:
        features (numpy.array): The valeus of each feature for all RSUs in network.
        W (list): Used to bias rewards between features

    Returns:
        int: Reward for all RSUs in RSU network
    """
    print("features",features)
    rewards = np.copy(features)
    # print(features)
    rewards[0] = np.nan_to_num(rewards[0],nan = -104)
    rewards[1] = np.divide(rewards[1],self.simulation_time[1]-self.simulation_time[0])
    rewards[0] = np.clip(rewards[0],-104,-50) # Clips decibels within expected range.
    rewards[1] = np.clip(rewards[1],0,500) # Clips number of messages/time step.
    rewards[0] = .25*np.sin(((rewards[0,:]+77)*np.pi)/54)+.25
    rewards[1] = .25*np.sin(((rewards[1,:]-250)*np.pi)/500)+.25

    return rewards, features

    

create_new_performance_model()
