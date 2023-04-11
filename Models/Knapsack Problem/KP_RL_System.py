from json import encoder
from tkinter import Variable
from typing import List, Tuple, Callable, Iterator
from collections import OrderedDict, deque
from itertools import combinations
import sys
import os

import torch
from torchviz import make_dot
import pandas as pd
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import IterableDataset
from scipy import stats,special

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from Actor_Critic import Actor_Critic
from KP_Datamodule import KP_Datamodule

import numpy as np
import argparse

np.set_printoptions(suppress=True)

class KP_System(LightningModule):
    def __init__(self, 
                     num_features: int = 2,
                     budget: int = 5,
                     nhead: int = 2,
                     num_layers: int = 8, 
                     lr: float = 1e-5,
                     model_directory = "/home/demo/",
                     save_data_bool:bool = False,
                     lmbda: float = 0.24098677879102673,
                     method: str = "sqrt"
                     ):

        super(KP_System,self).__init__()
        self.budget = budget
        self.nhead = nhead
        self.num_layers = num_layers
        self.lr = lr
        self.num_features = num_features

        self.actor_critic = Actor_Critic(num_features, nhead, self.num_layers)
        
        pd.options.display.float_format = '{:.4f}'.format
        # self.df_history = pd.DataFrame(columns=["intersection_idx","rsu_network","features","reward","critic_reward","entropy","actor_loss","critic_loss","loss"],dtype=object)
        self.df_history = pd.DataFrame(columns=["kp_pack","reward","loss"],dtype=object)
        self.df_new_data = self.df_history.copy()
        self.model_directory = model_directory
        self.model_history_file = os.path.join(self.model_directory,"model_history.csv")
        self.save_data_bool = save_data_bool
        self.lmbda = lmbda
        self.method = method

    def forward(self,items, budget) -> Tensor:
        items = torch.Tensor(items)[None,:]
        log_pointer_scores, _, kp_pack, critic_reward = self.actor_critic(items, budget)
        return log_pointer_scores, kp_pack

    def training_step(self,batch:Tuple[Tensor,Tensor]) -> OrderedDict:
        items = batch.type(torch.float)
        self.best_reward, self.best_pack = self.calculate_best_reward(items)
        log_pointer_scores, pointer_argmaxs, kp_pack, critic_reward = self.actor_critic(items, self.budget)
        ts_reward, loss = self.kp_loss(items, kp_pack[0,:])
        reward = torch.tensor(ts_reward, requires_grad=True,dtype=torch.float)

        padded_rewards = torch.zeros(pointer_argmaxs.shape[1],requires_grad=True)+.1
        # padded_rewards[rsu_idx != 0] = torch.tensor(rewards)
        padded_rewards[pointer_argmaxs[0,:] != 0] = reward.clone().detach().type(torch.float)

        loss, entropy, actor_loss, critic_loss = self.loss(log_pointer_scores, pointer_argmaxs, reward, padded_rewards, critic_reward)

        data = np.array((items.detach().numpy(),
                         np.around(padded_rewards.detach().numpy(),4),
                         loss.detach().numpy()),
                         dtype=object
                       )
        self.df_history.loc[self.df_history.shape[0]] = data
        self.df_new_data.loc[0] = data
        if self.save_data_bool:
            self.save_data()
        return loss

    def test_step(self,batch,batch_idx):
        items = batch.type(torch.float)
        self.best_reward, self.best_pack = self.calculate_best_reward(items)
        log_pointer_scores, pointer_argmaxs, kp_pack, critic_reward = self.actor_critic(items)
        ts_reward, _ = self.kp_loss(items, kp_pack[0,:])
        reward = torch.tensor(ts_reward, requires_grad=True,dtype=torch.float)
        padded_rewards = torch.zeros(pointer_argmaxs.shape[1],requires_grad=True)+.1
        # padded_rewards[rsu_idx != 0] = torch.tensor(rewards)
        padded_rewards[pointer_argmaxs[0,:] != 0] = reward.clone().detach().type(torch.float)

        loss, entropy, actor_loss, critic_loss = self.loss(log_pointer_scores, pointer_argmaxs, reward, padded_rewards, critic_reward)
        metrics = {"test_ts_reward": reward, "test_loss": loss}
        print(kp_pack)
        print(metrics)
        return metrics

    def calculate_best_reward(self,items):
        indices = np.arange(items.shape[1])
        path_combinations = list(combinations(indices, self.budget))
        best_reward = 0
        best_pack = None
        reward = np.empty(shape=(len(path_combinations)))
        reward = items[0,path_combinations,0]
        reward = reward.sum(axis=1)
        best_reward = reward.max()
        return best_reward, best_pack

    def kp_loss(self, items, kp_pack):
        kp_pack_reward = self.pack_reward(kp_pack, items)

        loss = kp_pack_reward - self.best_reward
        return kp_pack_reward, loss

    def pack_reward(self, pack, items):
        reward = 0
        reward = items[0,pack,0].sum()
        return reward

    def entropy(self,log_prob):
        """Function to calculate the losses for the actor and critic.

        Args:
            rewards (float): The actual rewards for the RSU network.
            critic_est (float): The estimate of the rewards for the RSU network from the critic.

        Returns:
            _type_: _description_
        """
        entropy = -log_prob.exp() * log_prob
        entropy = entropy.sum(1).mean()
        return entropy

    def actor_loss(self,log_prob,pointer_argmaxs,advs) -> torch.Tensor:
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

        log_prob_actions = log_prob[0,range(log_prob.shape[1]),pointer_argmaxs]
        log_prob_actions = -(advs * log_prob_actions)
        log_prob_actions = torch.nan_to_num(log_prob_actions,0.0)
        log_prob_actions = log_prob_actions[log_prob_actions!=0.0]
        loss = log_prob_actions.nanmean()

        return loss

    def critic_loss(self,rewards, critic_rewards) -> torch.Tensor:
        preprocessed_rewards = self.pre_process_reward_data(rewards,method = self.method)
        preprocessed_rewards = preprocessed_rewards[None,:]
        preprocessed_rewards = preprocessed_rewards.clone().detach().requires_grad_(True)
        return nn.MSELoss(reduction = 'mean')(preprocessed_rewards[0,:,:],critic_rewards)

    def loss(self,log_prob, pointer_argmaxs, rewards, padded_rewards, critic_rewards):
        # with torch.no_grad():
        #     advs = rewards - critic_rewards * rewards.std() + rewards.mean()
            # advs = (advs - advs.mean()) / advs.std()
            # targets = (rewards - rewards.mean()) / rewards.std()
        entropy = self.entropy(log_prob)
        actor_loss = self.actor_loss(log_prob, pointer_argmaxs, padded_rewards)
        # critic_loss = self.critic_loss(rewards, critic_rewards)
        critic_loss = torch.tensor(0)

        return actor_loss, entropy, actor_loss, critic_loss

    def pre_process_reward_data(self,reward_data,method = "boxcox",lmbda = 0.24098677879102673):
        if method == "boxcox":
            if isinstance(reward_data,torch.Tensor):
                reward_data = reward_data.detach().numpy()
            boxcox_rewards = stats.boxcox(reward_data+1e-8,lmbda=lmbda)
            return boxcox_rewards
        elif method == "sqrt":
            sqrt_rewards = torch.sqrt(reward_data)
            return sqrt_rewards
        elif method == "inv":
            inv_rewards = 1/(reward_data+1e-8)
            return inv_rewards
        elif method == "log":
            log_rewards = torch.log(reward_data+1e-8)
            return log_rewards
        else:
            return reward_data

    def pre_process_critic_reward_data(self,critic_reward_data,method = "boxcox",lmbda = 0.24098677879102673):
        if method == "boxcox":
            if isinstance(critic_reward_data,torch.Tensor):
                critic_reward_data = critic_reward_data.detach().numpy()
            inv_boxcox_critic_rewards = special.inv_boxcox(critic_reward_data,lmbda)
            return inv_boxcox_critic_rewards
        elif method == "sqrt":
            inv_sqrt_critic_rewards = torch.pow(critic_reward_data,2)
            return inv_sqrt_critic_rewards
        elif method == "inv":
            inv_critic_rewards = 1/(critic_reward_data+1e-8)
            return inv_critic_rewards
        elif method == "log":
            inv_log_critic_rewards = torch.exp(critic_reward_data)
            return inv_log_critic_rewards
        else:
            return critic_reward_data
     
    def configure_optimizers(self) -> List[Optimizer]:
        """Initializes Adam optimizers for actor and critic.

        Returns:
            List[Optimizer]: Optimizers for actor and critic.
        """
        optimizer = Adam(self.actor_critic.parameters(), lr=self.lr)
        return optimizer

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
    max_epochs = 10
    save_model_bool = False
    display_figures = True

    trainer = Trainer(max_epochs = max_epochs)
    directory_path = "/home/demo/RSU_RL_Placement/trained_models/"
    model_name = "kp_2000_epochs_10_items_5_budget"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_name = "kp_test"
    checkpoint_directory = os.path.join(directory_path,checkpoint_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,checkpoint_name)

    if save_model_bool:
            os.makedirs(model_directory)
            f = open(os.path.join(model_directory,"output.txt"),'w')
            sys.stdout = f

    model = KP_System(model_directory = model_directory, save_data_bool= save_model_bool, budget = 5)
    trainer = Trainer(max_epochs = max_epochs)
    datamodule = KP_Datamodule(n_scenarios=100, n_items=10)
    trainer.fit(model,datamodule=datamodule)
    if save_model_bool:
        save_model(model,model_directory,model_path)

    y1, y2 = model(next(iter(datamodule.database)), budget = 5)
    make_dot(y1, params=dict(list(model.named_parameters()))).render("KP_RL_torchviz", format="png")

