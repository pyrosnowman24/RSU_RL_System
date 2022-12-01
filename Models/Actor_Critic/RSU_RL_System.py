from cgi import test
from cmath import isnan
from ctypes import pointer
from json import encoder
from tkinter import Variable
from typing import List, Tuple, Callable, Iterator
from collections import OrderedDict, deque, namedtuple
import sys
import os

import torch
import pandas as pd
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import IterableDataset
from scipy import stats,special

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent
from Actor_Critic import Actor_Critic
from Models.RSU_Intersections_Datamodule import RSU_Intersection_Datamodule

import numpy as np
import argparse

np.set_printoptions(suppress=True)

class DRL_System(LightningModule):
    def __init__(self, 
                     agent,
                     num_features: int = 4,
                     nhead: int = 4,
                     num_layers: int = 1, 
                     lr: float = 5e-4,
                     W: int = [.5,.5],
                     model_directory = "/home/acelab/",
                     save_data_bool:bool = False,
                     use_sim: bool = True,
                     rsu_performance_df = None,
                     lmbda: float = 0.24098677879102673,
                     method: str = "sqrt"
                     ):

        super(DRL_System,self).__init__()
        self.agent = agent
        self.nhead = nhead
        self.num_layers = num_layers
        self.lr = lr
        self.W = W
        self.num_features = num_features

        self.actor_critic = Actor_Critic(num_features, nhead, self.W, self.num_layers)
        
        pd.options.display.float_format = '{:.4f}'.format
        # self.df_history = pd.DataFrame(columns=["intersection_idx","rsu_network","features","reward","critic_reward","entropy","actor_loss","critic_loss","loss"],dtype=object)
        self.df_history = pd.DataFrame(columns=["intersection_idx","rsu_network","reward","critic_reward","entropy","actor_loss","critic_loss","loss"],dtype=object)
        self.df_new_data = self.df_history.copy()
        self.model_directory = model_directory
        self.model_history_file = os.path.join(self.model_directory,"model_history.csv")
        self.save_data_bool = save_data_bool
        self.use_sim = use_sim
        self.rsu_performance_df = rsu_performance_df
        self.lmbda = lmbda
        self.method = method

    def forward(self,intersections: Tensor, mask: Tensor) -> Tensor:
        _, _, rsu_network, critic_reward = self.actor_critic(intersections, mask)
        return rsu_network, critic_reward

    def training_step(self,batch:Tuple[Tensor,Tensor],batch_idx) -> OrderedDict:
        intersections = batch[0]
        intersection_idx = batch[1]
        rsu_network_idx = batch[2]
        mask = batch[3]
        # intersections = intersections[:,mask[0,:],:]
        log_pointer_scores, pointer_argmaxs, rsu_idx, critic_reward = self.actor_critic(intersections[:,:,1:],mask)

        # This is what I need to fix tomorrow
        if self.use_sim:
            rewards, features = self.agent.simulation_step(rsu_idx,intersection_idx[0],model = "Q Learning Positive")
            rewards = torch.tensor([np.array(rewards)],requires_grad=True,dtype=torch.float)
        else:
            rsu_intersections = intersections[0,rsu_idx,:].detach().numpy()
            rewards = np.empty(shape=(1,rsu_intersections.shape[0]))
            for i,intersection in enumerate(rsu_intersections):
                info = self.rsu_performance_df[int(intersection[0]) == self.rsu_performance_df[:,0].astype(int)]
                rewards[:,i] = info[0,-1]
            rewards = torch.tensor(rewards, requires_grad=True,dtype=torch.float)
        padded_rewards = torch.zeros(pointer_argmaxs.shape[1],requires_grad=True)+.1
        # padded_rewards[rsu_idx != 0] = torch.tensor(rewards)
        padded_rewards[pointer_argmaxs[0,:] != 0] = torch.tensor(rewards.clone().detach(),dtype=torch.float)

        loss, entropy, actor_loss, critic_loss = self.loss(log_pointer_scores, pointer_argmaxs, rewards, padded_rewards, critic_reward)

        print("loss",loss)

        # data = np.array((intersections.detach().numpy(),rsu_idx,features,rewards.detach().numpy(), critic_reward.detach().numpy(), entropy.detach().numpy(), actor_loss.detach().numpy(), critic_loss.detach().numpy(), loss.detach().numpy()),dtype=object)
        preprocessed_critic_rewards = self.pre_process_critic_reward_data(critic_reward, method = self.method)
        data = np.array((intersections.detach().numpy(),rsu_idx,np.around(padded_rewards.detach().numpy(),4), np.around(preprocessed_critic_rewards.detach().numpy(),4), entropy.detach().numpy(), actor_loss.detach().numpy(), critic_loss.detach().numpy(), loss.detach().numpy()),dtype=object)
        self.df_history.loc[self.df_history.shape[0]] = data
        self.df_new_data.loc[0] = data
        if self.save_data_bool:
            self.save_data()
        return loss

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
        preprocessed_rewards = torch.tensor(preprocessed_rewards,requires_grad=True)
        return nn.MSELoss(reduction = 'mean')(preprocessed_rewards,critic_rewards)

    def loss(self,log_prob, pointer_argmaxs, rewards, padded_rewards, critic_rewards):
        # with torch.no_grad():
        #     advs = rewards - critic_rewards * rewards.std() + rewards.mean()
            # advs = (advs - advs.mean()) / advs.std()
            # targets = (rewards - rewards.mean()) / rewards.std()
        entropy = self.entropy(log_prob)
        actor_loss = self.actor_loss(log_prob, pointer_argmaxs, padded_rewards)
        critic_loss = self.critic_loss(rewards, critic_rewards)

        return actor_loss + critic_loss - entropy, entropy, actor_loss, critic_loss

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
    max_epochs = 50
    train_new_model = True
    save_model_bool = False
    display_figures = True
    use_sim = False

    simulation_agent = Agent()
    trainer = Trainer(max_epochs = max_epochs)
    directory_path = "/home/acelab/Dissertation/RSU_RL_Placement/trained_models/"
    model_name = "50_critic_no_buffer"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)
    
    if not use_sim: rsu_performance_df = pd.read_csv("/home/acelab/Dissertation/RSU_RL_Placement/rsu_performance_dataset").to_numpy()
    else: rsu_performance_df = None

    checkpoint_name = "Training_for_Reward"
    checkpoint_directory = os.path.join(directory_path,checkpoint_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,checkpoint_name)

    if save_model_bool:
            os.makedirs(model_directory)
            f = open(os.path.join(model_directory,"output.txt"),'w')
            sys.stdout = f
    if train_new_model:
        model = DRL_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool, use_sim = use_sim, rsu_performance_df = rsu_performance_df)
        trainer = Trainer(max_epochs = max_epochs)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)

    else:
        model = DRL_System(simulation_agent,model_directory = model_directory, save_data_bool= save_model_bool)
        model.load_state_dict(torch.load(checkpoint_path))
        trainer = Trainer(max_epochs = max_epochs,gradient_clip_value = 0.5)
        datamodule = RSU_Intersection_Datamodule(simulation_agent)
        trainer.fit(model,datamodule=datamodule)
        if save_model_bool:
            save_model(model,model_directory,model_path)
