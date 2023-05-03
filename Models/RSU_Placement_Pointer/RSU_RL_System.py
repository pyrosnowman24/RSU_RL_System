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

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent
from Models.RSU_Placement_Pointer.Actor_Critic_ptr import Actor_Critic
from Models.RSU_Placement_Pointer.RSU_Intersection_Datamodule_Fast import RSU_Intersection_Datamodule
from Models.Knapsack_Algorithm.KA_RSU import KA_RSU
from Models.GA.GA_RSU import GA_RSU


import numpy as np
import argparse

np.set_printoptions(suppress=True)

class RSU_Placement_System(LightningModule):
    def __init__(self, 
                     num_features: int = 2,
                     hidden_size: int = 256,
                     lr: float = 1e-3,
                     model_directory = "/home/demo/",
                     save_data_bool:bool = False,
                     ):

        super(RSU_Placement_System,self).__init__()
        self.hidden_size = hidden_size
        self.lr = lr
        self.num_features = num_features

        self.actor_critic = Actor_Critic(num_features, hidden_size)
        
        pd.options.display.float_format = '{:.4f}'.format
        self.df_validation_history = pd.DataFrame(columns=["validation_loss"],dtype=object)
        self.df_validation_history_new = self.df_validation_history.copy()
        self.df_history = pd.DataFrame(columns=["kp_pack","reward","weight","budget","loss"],dtype=object)
        self.df_new_data = self.df_history.copy()
        self.model_directory = model_directory
        self.model_history_file = os.path.join(self.model_directory,"model_history.csv")
        self.validation_history_file = os.path.join(self.model_directory,"validation_history.csv")
        self.save_data_bool = save_data_bool
        self.knapsack_algorithm = KA_RSU()
        self.genetic_algorithm = GA_RSU(population_size = 50, mutation_prob = .1)

    def forward(self,intersections, budget) -> Tensor:
        intersections = torch.Tensor(intersections)[None,:]
        intersections_normalized = self.normalize_items(intersections.detach().clone(), budget)
        log_pointer_scores, pointer_argmaxs, kp_pack = self.actor_critic(intersections_normalized, budget)
        reward, weight = self.pack_reward(kp_pack, intersections)
        return log_pointer_scores, kp_pack, reward, weight

    def training_step(self,batch:Tuple[Tensor,Tensor]) -> OrderedDict:
        intersections = batch[0].type(torch.float)
        intersection_idxs = batch[1]
        self.budget = batch[2].numpy()[0]
        self.best_pack = batch[3].numpy()[0]
        self.best_reward = batch[4][0]
        # print(self.best_reward, self.best_pack)

        inputs = torch.empty(size = (1,intersections.shape[1],2))
        inputs = intersections[:,:,-2:]

        inputs = self.normalize_items(inputs, self.budget)

        # self.best_reward, self.best_pack = self.calculate_best_reward_brute_force(inputs)
        # print(self.best_reward, self.best_pack)
        # self.best_reward, self.best_weight, self.best_pack = self.calculate_best_reward_knapsack(inputs)
        # print(self.best_reward, self.best_weight, self.best_pack)
        # self.best_reward, self.best_weight, self.best_pack = self.calculate_best_reward_genetic(inputs)
        # print(self.best_reward, self.best_weight, self.best_pack)

        log_pointer_scores, pointer_argmaxs, kp_pack = self.actor_critic(inputs, self.budget)
        # print(kp_pack)
        # print("kp pack",kp_pack)
        # print("best pack",self.best_pack)
        # print("budget", self.budget)
        # print(log_pointer_scores)
        loss = self.nll_loss(log_pointer_scores,kp_pack)
        # print("loss",loss)
        reward, weight = self.pack_reward(kp_pack, inputs)

        data = np.array((kp_pack.detach().numpy(),
                         reward.detach().numpy(),
                         weight.detach().numpy(),
                         self.budget,
                         loss.detach().item()),
                         dtype=object
                       )
        self.df_history.loc[self.df_history.shape[0]] = data
        self.df_new_data.loc[0] = data
        if self.save_data_bool:
            self.save_data()
        # print(self.budget)
        self.log("loss", loss)
        # print('\n')
        return loss
    
    def validation_step(self, batch:Tuple[Tensor,Tensor], batch_idx):
        intersections = batch[0].type(torch.float)
        intersection_idxs = batch[1]
        self.budget = batch[2].numpy()[0]
        self.best_pack = batch[3].numpy()[0]
        self.best_reward = batch[4][0]

        inputs = torch.empty(size = (1,intersections.shape[1],2))
        inputs = intersections[:,:,-2:]

        inputs = self.normalize_items(inputs, self.budget)

        # self.best_reward, self.best_weight, self.best_pack = self.calculate_best_reward_knapsack(inputs)
        # self.best_reward, self.best_weight, self.best_pack = self.calculate_best_reward_genetic(inputs)

        log_pointer_scores, pointer_argmaxs, kp_pack = self.actor_critic(inputs, self.budget)
        loss = self.nll_loss(log_pointer_scores,kp_pack)
        self.log("val_loss", loss, on_epoch=True)

        data = np.array(loss.detach().numpy())
        self.df_validation_history.loc[self.df_validation_history.shape[0]] = data
        self.df_validation_history_new.loc[0] = data

        if self.save_data_bool:
            self.save_validation()

    def calculate_best_reward_brute_force(self,intersections, budget):
        indices = np.arange(intersections.shape[1])
        best_reward = 0
        best_pack = None
        for i in range(1,budget+1):
            path_combinations = []
            path_combinations.extend(list(combinations(indices, i)))
            reward = np.empty(shape=(len(path_combinations)))
            reward = intersections[0,path_combinations,0]
            reward = reward.sum(axis=1)
            best_reward_i = reward.max()
            best_reward_index = reward.argmax()
            if best_reward_i > best_reward:
                best_reward = best_reward_i.clone()
                best_pack = path_combinations[best_reward_index]
        best_pack = np.asarray(best_pack)
        # best_items = intersections[0,best_pack,:]
        # diffs = best_items[:,0] - best_items[:,1]
        # diffs, indexes = torch.sort(diffs, descending=True)
        # best_pack = best_pack[indexes]
        return best_reward, best_pack
    
    def calculate_best_reward_knapsack(self,intersections):
        return self.knapsack_algorithm.ka_algorithm(intersections, self.budget)
    
    def calculate_best_reward_genetic(self,intersections):
        return self.genetic_algorithm(intersections=intersections[0,:,:], epochs=50, budget = self.budget)

    def nll_loss(self,log_pointer_scores, pack_idx):
        if self.best_pack.shape[0] > pack_idx.shape[0]:
            reshaped_log_pointer_score = log_pointer_scores[:self.best_pack.shape[0], :].clone()
            padded_best_pack = torch.tensor(self.best_pack+1)
        else:
            padded_best_pack = torch.zeros(pack_idx.shape[0])
            padded_best_pack[:self.best_pack.shape[0]] = torch.tensor(self.best_pack+1)
            reshaped_log_pointer_score = log_pointer_scores[:pack_idx.shape[0], :].clone()
        loss =  torch.nn.NLLLoss()(reshaped_log_pointer_score, padded_best_pack.type(torch.long))
        return loss

    def pack_reward(self, pack, intersections):
        reward = 0
        reward = intersections[0,pack,0].sum()
        cost = 0
        cost = intersections[0,pack,1].sum()
        return reward, cost
    
    def normalize_items(self,intersections,budget):
        intersections[0,:,0] = torch.divide(intersections[:,:,0],torch.max(intersections[:,:,0]))
        intersections[0,:,1] = torch.divide(intersections[:,:,1],torch.tensor(budget))
        return intersections

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

    def save_validation(self):
        if not os.path.isfile(self.validation_history_file):
            self.df_validation_history_new.to_csv(self.validation_history_file, index=False)
        else: # else it exists so append without writing the header
            self.df_validation_history_new.to_csv(self.validation_history_file, index=False, mode='a', header=False)

def save_model(model,model_directory,model_path):
        model_history_file = os.path.join(model_directory,"model_history.csv")
        torch.save(model.state_dict(),model_path)
        
if __name__ == '__main__':
    max_epochs = 500
    save_model_bool = True
    
    trainer = Trainer(max_epochs = max_epochs)
    directory_path = "/home/demo/RSU_RL_Placement/trained_models/"
    model_name = "knapsack_300_epochs_100_scenarios_final5"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_name = "kp_test"
    checkpoint_directory = os.path.join(directory_path,checkpoint_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,checkpoint_name)

    if save_model_bool:
            os.makedirs(model_directory)
            # f = open(os.path.join(model_directory,"output.txt"),'w')
            # sys.stdout = f
    simulation_agent = Agent()
    knapsack_algorithm = KA_RSU()
    model = RSU_Placement_System(model_directory=model_directory, save_data_bool=save_model_bool, lr = 1e-4)
    trainer = Trainer(max_epochs = max_epochs)
    genetic_algorithm = GA_RSU(population_size = 50, mutation_prob = .1)
    knapsack_algorithm = KA_RSU()
    datamodule = RSU_Intersection_Datamodule(simulation_agent, knapsack_algorithm, n_scenarios=100, min_budget=5, max_budget=6, min_intersections = 10, max_intersections = 25, min_weight=1, max_weight=3)
    # datamodule = RSU_Intersection_Datamodule(simulation_agent, n_scenarios=100, min_budget=5, max_budget=6, min_intersections = 10, max_intersections = 20)
    trainer.fit(model,datamodule=datamodule)
    trainer.validate(model,datamodule=datamodule)
    if save_model_bool:
        save_model(model,model_directory,model_path)

    # y1, y2 = model(next(iter(datamodule.database)), budget = 5)
    # make_dot(y1, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True).render("KPB_RL_torchviz_show_attrs", format="png")