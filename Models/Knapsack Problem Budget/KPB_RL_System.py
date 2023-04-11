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

from Actor_Critic_ptr import Actor_Critic
from KPB_Datamodule import KPB_Datamodule

import numpy as np
import argparse

np.set_printoptions(suppress=True)

class KPB_System(LightningModule):
    def __init__(self, 
                     num_features: int = 2,
                     budget: int = 5,
                     hidden_size: int = 256,
                     num_layers: int = 8, 
                     lr: float = 1e-5,
                     model_directory = "/home/demo/",
                     save_data_bool:bool = False,
                     lmbda: float = 0.24098677879102673,
                     method: str = "sqrt"
                     ):

        super(KPB_System,self).__init__()
        self.budget = budget
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.num_features = num_features

        self.actor_critic = Actor_Critic(num_features, hidden_size)
        
        pd.options.display.float_format = '{:.4f}'.format
        # self.df_history = pd.DataFrame(columns=["intersection_idx","rsu_network","features","reward","critic_reward","entropy","actor_loss","critic_loss","loss"],dtype=object)
        self.df_history = pd.DataFrame(columns=["kp_pack","reward","weight","loss"],dtype=object)
        self.df_new_data = self.df_history.copy()
        self.model_directory = model_directory
        self.model_history_file = os.path.join(self.model_directory,"model_history.csv")
        self.save_data_bool = save_data_bool
        self.lmbda = lmbda
        self.method = method

    def forward(self,items, budget) -> Tensor:
        items = torch.Tensor(items)[None,:]
        log_pointer_scores, pointer_argmaxs, kp_pack, critic_reward = self.actor_critic(items, budget)
        return log_pointer_scores, kp_pack

    def training_step(self,batch:Tuple[Tensor,Tensor]) -> OrderedDict:
        items = batch.type(torch.float)
        items[:,:,1] = torch.divide(items[:,:,1],self.budget)
        items[:,:,0] = torch.divide(items[:,:,0],torch.max(items[:,:,0]))
        self.best_reward, self.best_pack = self.calculate_best_reward(items)
        log_pointer_scores, pointer_argmaxs, kp_pack, critic_reward = self.actor_critic(items, self.budget)
        print("kp pack",kp_pack)
        print("best pack",self.best_pack)
        # print(log_pointer_scores)
        loss = self.nll_loss(log_pointer_scores,kp_pack)
        print("loss",loss)
        reward, weight = self.pack_reward(kp_pack, items)

        data = np.array((kp_pack.detach().numpy(),
                         reward.detach().numpy(),
                         weight.detach().numpy(),
                         loss.detach().item()),
                         dtype=object
                       )
        self.df_history.loc[self.df_history.shape[0]] = data
        self.df_new_data.loc[0] = data
        if self.save_data_bool:
            self.save_data()
        return loss

    def test_step(self,batch,batch_idx):
        items = batch.type(torch.float)
        # self.best_reward, self.best_pack = self.calculate_best_reward(items)
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
        best_reward = 0
        best_pack = None
        for i in range(1,self.budget+1):
            path_combinations = []
            path_combinations.extend(list(combinations(indices, i)))
            reward = np.empty(shape=(len(path_combinations)))
            reward = items[0,path_combinations,0]
            reward = reward.sum(axis=1)
            best_reward_i = reward.max()
            best_reward_index = reward.argmax()
            if best_reward_i > best_reward:
                best_reward = best_reward_i.clone()
                best_pack = path_combinations[best_reward_index]
        best_pack = np.asarray(best_pack)
        best_items = items[0,best_pack,:]
        # diffs = best_items[:,0] - best_items[:,1]
        # diffs, indexes = torch.sort(diffs, descending=True)
        # best_pack = best_pack[indexes]
        return best_reward, best_pack

    # def kp_loss(self, items, kp_pack):
    #     items = self.normalize_items(items)
    #     kp_pack_reward, kp_pack_cost = self.pack_reward(kp_pack, items)
    #     space_violation = torch.maximum(kp_pack_cost-self.budget,torch.zeros(1))
    #     var1 = 4
    #     loss = -kp_pack_reward + (var1 * space_violation) + 5
    #     return kp_pack_reward, kp_pack_cost, loss

    def kp_loss(self, items, kp_pack):
        kp_pack_reward, kp_pack_reward = self.pack_reward(kp_pack, items)
        loss = (kp_pack_reward - self.best_reward)
        if (len(kp_pack) - self.budget) > 0 or len(kp_pack) == 0:
            loss = loss + self.budget
        return kp_pack_reward, kp_pack_reward, loss

    def nll_loss(self,log_pointer_scores, pack_idx):
        if self.best_pack.shape[0] > pack_idx.shape[0]:
            reshaped_log_pointer_score = log_pointer_scores[:self.best_pack.shape[0], :].clone()
            padded_best_pack = torch.tensor(self.best_pack+1)
        else:
            padded_best_pack = torch.zeros(pack_idx.shape[0])
            padded_best_pack[:self.best_pack.shape[0]] = torch.tensor(self.best_pack+1)
            reshaped_log_pointer_score = log_pointer_scores[:pack_idx.shape[0], :].clone()
        loss =  torch.nn.NLLLoss()(reshaped_log_pointer_score, padded_best_pack.type(torch.long)) # Maybe something weird with this, look into it more
        return loss

    def pack_reward(self, pack, items):
        reward = 0
        reward = items[0,pack,0].sum()
        cost = 0
        cost = items[0,pack,1].sum()
        return reward, cost
    
    def normalize_items(self,items):
        items[0,:,0] = torch.divide(items[0,:,0],torch.max(items[0,:,0]))
        items[0,:,1] = torch.divide(items[0,:,1],torch.tensor(self.budget))
        return items

    def configure_optimizers(self) -> List[Optimizer]:
        """Initializes Adam optimizers for actor and critic.

        Returns:
            List[Optimizer]: Optimizers for actor and critic.
        """
        optimizer = Adam(self.actor_critic.parameters(), lr=self.lr)
        return optimizer

    def save_data(self):
        if not os.path.isfile(self.model_history_file):
            self.df_new_data.to_csv(self.model_history_file, index=False)
        else: # else it exists so append without writing the header
            self.df_new_data.to_csv(self.model_history_file, index=False, mode='a', header=False)

def save_model(model,model_directory,model_path):
        model_history_file = os.path.join(model_directory,"model_history.csv")
        torch.save(model.state_dict(),model_path)
        
if __name__ == '__main__':
    max_epochs = 2000
    save_model_bool = True
    display_figures = True

    trainer = Trainer(max_epochs = max_epochs)
    directory_path = "/home/demo/RSU_RL_Placement/trained_models/"
    # model_name = "kpb_2000_epochs_20_items_5_budget"
    model_name = "test2"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_name = "kp_test"
    checkpoint_directory = os.path.join(directory_path,checkpoint_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,checkpoint_name)

    if save_model_bool:
            os.makedirs(model_directory)
            # f = open(os.path.join(model_directory,"output.txt"),'w')
            # sys.stdout = f

    model = KPB_System(model_directory = model_directory, save_data_bool= save_model_bool, budget = 5)
    trainer = Trainer(max_epochs = max_epochs)
    datamodule = KPB_Datamodule(n_scenarios=100, n_items=15)



    trainer.fit(model,datamodule=datamodule)
    if save_model_bool:
        save_model(model,model_directory,model_path)

    # y1, y2 = model(next(iter(datamodule.database)), budget = 5)
    # make_dot(y1, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True).render("KPB_RL_torchviz_show_attrs", format="png")