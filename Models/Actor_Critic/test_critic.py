import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import IterableDataset
from scipy import stats,special

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent
from Models.RSU_Intersections_Datamodule import RSU_Intersection_Datamodule

import numpy as np
import argparse

class Embedding(nn.Module):
    def __init__(self,c_inputs,c_embed):
        super(Embedding,self).__init__()
        self.embedding_layer = nn.Linear(c_inputs, c_embed, bias=False)

    def forward(self,data):
        return self.embedding_layer(data)

class Critic(nn.Module):
    def __init__(self,num_features = 4,
                      c_embed = 16 ):
        super(Critic,self).__init__()
        self.embedding = Embedding(num_features,c_embed)
        self.network = nn.Sequential(nn.Linear(c_embed,32),
                                     nn.ReLU(),
                                     nn.Dropout(p = .2),
                                     nn.Linear(32,64),
                                     nn.ReLU(),
                                     nn.Linear(64,64),
                                     nn.ReLU(),
                                     nn.Linear(64,64),
                                     nn.ReLU(),
                                     nn.Linear(64,64),
                                     nn.ReLU(),
                                     nn.Linear(64,32),
                                     nn.ReLU(),
                                     nn.Linear(32,1)
                                     )
        

    def forward(self,intersections):
        embedded_state = self.embedding(intersections)

        rewards = []
        for sample in embedded_state:
            reward = self.network(sample)
            rewards.append(reward)
        rewards = torch.stack(rewards,dim=0)

        return rewards[:,:,0]

class Critic_RF():
    def __init__(self, 
                 num_features = 4,
                 n_estimators = 100,
                 max_depth = 100,
                 min_samples_split = 50,
                 random_state = 1
                 ):
        # self.rf_regressor = MultiOutputRegressor(
        #     RandomForestRegressor(n_estimators= n_estimators)
        # )
        self.rf_regressor = RandomForestRegressor(n_estimators= n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)

    def fit_data(self, rsu_performance_df):
        features = np.empty(shape=(rsu_performance_df.shape[0],4)) 
        rewards = np.empty(shape=(rsu_performance_df.shape[0],))
        for i,data in enumerate(rsu_performance_df):
            features[i,:] = data[:4]
            rewards[i] = data[-1]
        self.rf_regressor = RandomizedSearchCV(self.rf_regressor, n_iter=10,
                        param_distributions = {'max_depth': range(1, 15),
                                               'min_samples_split': range(2, 50)},
                        cv=5, n_jobs=-1, random_state=3,
                        scoring='neg_mean_absolute_error')
        self.rf_regressor = self.rf_regressor.fit(features, rewards)
        

    def forward(self,features):
        return self.rf_regressor.predict(features)

class Critic_Intersections(LightningModule):
    def __init__(self,
                 rsu_performance_df,
                 num_features: int = 4,
                 lr: float = .00001,
                 ):
        super(Critic_Intersections,self).__init__()

        self.critic = Critic(num_features)

        self.rsu_performance_df = rsu_performance_df
        self.rewards_df = self.rsu_performance_df[:,-1].astype(float).reshape(-1,1)
        self.rewards_df = torch.tensor(self.rewards_df, requires_grad=False,dtype=torch.float)
        self.rewards_df = self.pre_process_reward_data(self.rewards_df,method = "sqrt")
        scaler = StandardScaler()
        scaler.fit(self.rewards_df)
        self.rewards_df = scaler.transform(self.rewards_df)

        # Add the scaling to the preprocessing
        # removing the call to preprocessing in the training/prediction functions

        fig,ax = plt.subplots(1)
        ax.hist(self.rewards_df)
        plt.show()
        quit()

        self.lr = lr

        self.predict_reward_history = np.empty(0)
        self.predict_criticReward_history = np.empty(0)

    def forward(self,intersection):
        rewards = self.critic(intersection)
        return rewards

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        intersections = batch[0][:,1:,:]
        mask = batch[3][0,1:].detach().numpy()
        rsu_intersections = np.empty(shape=(0,5))
        for i,value in enumerate(mask):
            if value:
                rsu_intersections = np.vstack([rsu_intersections,intersections[:,i,:]])
        rsu_intersections = torch.tensor(rsu_intersections[None,:],requires_grad=True,dtype=torch.float)

        critic_rewards = self.critic(rsu_intersections[:,:,1:])

        rewards = np.empty(shape=(1,rsu_intersections.shape[1]))
        for i,intersection in enumerate(rsu_intersections[0,:,:].detach().numpy()):
            info = self.rsu_performance_df[int(intersection[0]) == self.rsu_performance_df[:,0].astype(int)]
            rewards[:,i] = info[0,-1]
        rewards = torch.tensor(rewards, requires_grad=True,dtype=torch.float)
        preprocessed_rewards = self.pre_process_reward_data(rewards,method = "sqrt")

        self.predict_criticReward_history = np.append(self.predict_criticReward_history, critic_rewards)
        self.predict_reward_history = np.append(self.predict_reward_history, preprocessed_rewards)

        print("Rewards",preprocessed_rewards)
        print("Predicted Rewards",critic_rewards)

        return critic_rewards

    def plot_predict_performance(self):
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.boxplot(self.predict_criticReward_history)
        ax2.boxplot(self.predict_reward_history)
        ax1.set_title("Critic Rewards")
        ax2.set_title("Rewards")
        plt.show()

    def training_step(self,batch,batch_idx):
        intersections = batch[0][:,1:,:]
        intersection_idx = batch[1]
        rsu_network_idx = batch[2]
        mask = batch[3][0,1:].detach().numpy()
        rsu_intersections = np.empty(shape=(0,5))
        for i,value in enumerate(mask):
            if value:
                rsu_intersections = np.vstack([rsu_intersections,intersections[:,i,:]])
        rsu_intersections = torch.tensor(rsu_intersections[None,:],requires_grad=True,dtype=torch.float)

        critic_rewards = self.critic(rsu_intersections[:,:,1:])

        rewards = np.empty(shape=(1,rsu_intersections.shape[1]))
        for i,intersection in enumerate(rsu_intersections[0,:,:].detach().numpy()):
            rewards[:,i] = self.rewards_df[int(intersection[0]) == self.rsu_performance_df[:,0].astype(int)]
        rewards = torch.tensor(rewards, requires_grad=True,dtype=torch.float)
        
        bad_intersection_mask = rewards>.01
        rewards_masked = rewards[bad_intersection_mask]
        critic_rewards_masked = critic_rewards[bad_intersection_mask]

        loss = self.loss(rewards_masked, critic_rewards_masked)

        return loss

    def loss(self, rewards, critic_rewards):
        preprocessed_rewards = self.pre_process_reward_data(rewards,method = "sqrt")
        # print("Rewards",preprocessed_rewards)
        # print("Predicted Rewards",critic_rewards)
        return nn.L1Loss(reduction = 'mean')(critic_rewards,preprocessed_rewards)

    def configure_optimizers(self):
        optimizer = Adam(self.critic.parameters(), lr=self.lr)
        return optimizer

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
            inv_sqrt_critic_rewards = torch.power(critic_reward_data,2)
            return inv_sqrt_critic_rewards
        elif method == "inv":
            inv_critic_rewards = 1/(critic_reward_data+1e-8)
            return inv_critic_rewards
        elif method == "log":
            inv_log_critic_rewards = torch.exp(critic_reward_data)
            return inv_log_critic_rewards
        else:
            return critic_reward_data

class Critic_Intersections_RF(LightningModule):
    def __init__(self,
                 rsu_performance_df,
                 num_features: int = 4,
                 lr: float = .00001,
                 ):
        super(Critic_Intersections_RF,self).__init__()

        # self.critic = Critic(num_features)
        self.critic = Critic_RF()
        self.critic.fit_data(rsu_performance_df)
        self.rsu_performance_df = rsu_performance_df
        self.lr = lr

    def forward(self,intersection):
        rewards = self.critic.forward(intersection)
        return rewards

    def training_step(self,batch,batch_idx):
        intersections = batch[0][:,1:,:]
        mask = batch[3][0,1:].detach().numpy()
        rsu_intersections = np.empty(shape=(0,5))
        for i,value in enumerate(mask):
            if value:
                rsu_intersections = np.vstack([rsu_intersections,intersections[:,i,:]])
        rsu_intersections = rsu_intersections[None,:]

        rewards = np.empty(shape=(1,rsu_intersections.shape[1]))
        for i,intersection in enumerate(rsu_intersections[0,:,:]):
            info = self.rsu_performance_df[int(intersection[0]) == self.rsu_performance_df[:,0].astype(int)]
            rewards[:,i] = info[0,-1]

        print(rewards)
        critic_rewards = self.critic.forward(rsu_intersections[0,:,1:])
        print(critic_rewards)
        print(self.critic.rf_regressor.best_params_)
        print(-self.critic.rf_regressor.best_score_)       
        quit()

        # print(np.rot90(rewards))
        print(np.rot90(critic_rewards))
        
        loss =  mean_squared_error(rewards,critic_rewards)
        return torch.tensor(loss, requires_grad=True)
        
    def validation_step(self,batch):
        intersections = batch[0][:,1:,:]
        mask = batch[3][0,1:].detach().numpy()
        rsu_intersections = np.empty(shape=(0,5))
        for i,value in enumerate(mask):
            if value:
                rsu_intersections = np.vstack([rsu_intersections,intersections[:,i,:]])
        rsu_intersections = rsu_intersections[None,:]

        rewards = np.empty(shape=(1,rsu_intersections.shape[1]))
        for i,intersection in enumerate(rsu_intersections[0,:,:]):
            info = self.rsu_performance_df[int(intersection[0]) == self.rsu_performance_df[:,0].astype(int)]
            rewards[:,i] = info[0,-1]
        rewards = np.rot90(rewards,3)

        critic_rewards = self.critic.forward(rsu_intersections[0,:,1:])

        loss =  mean_squared_error(rewards,critic_rewards)
        return torch.tensor(loss, requires_grad=True)

    def loss(self, rewards, critic_rewards):
        preprocessed_rewards = self.pre_process_reward_data(rewards,method = "sqrt")
        print(preprocessed_rewards)
        print(critic_rewards)
        return nn.L1Loss(reduction = 'mean')(critic_rewards,preprocessed_rewards)

    def configure_optimizers(self):
        # optimizer = Adam(self.critic.parameters(), lr=self.lr)
        # return optimizer
        return None

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
            inv_sqrt_critic_rewards = torch.power(critic_reward_data,2)
            return inv_sqrt_critic_rewards
        elif method == "inv":
            inv_critic_rewards = 1/(critic_reward_data+1e-8)
            return inv_critic_rewards
        elif method == "log":
            inv_log_critic_rewards = torch.exp(critic_reward_data)
            return inv_log_critic_rewards
        else:
            return critic_reward_data

if __name__ == '__main__':
    max_epochs = 300
    rsu_performance_df = pd.read_csv("/home/acelab/Dissertation/RSU_RL_Placement/rsu_performance_dataset").to_numpy()
    trainer = Trainer(max_epochs = max_epochs)
    simulation_agent = Agent()

    model = Critic_Intersections(rsu_performance_df)
    trainer = Trainer(max_epochs = max_epochs)
    datamodule = RSU_Intersection_Datamodule(simulation_agent)
    trainer.fit(model,datamodule=datamodule)

    trainer.predict(model,datamodule=datamodule)

    model.plot_predict_performance()



# class test():
#     def __init__(self):
#         self.rf_regressor = RandomForestRegressor()
#         num = 2000
#         self.x = np.random.uniform(0,100,size = (num,2))
#         self.y = np.random.uniform(0,100,size = (num,))
#         self.x_small = np.random.uniform(0,100,size = (10,2))
#         self.y_small = np.random.uniform(0,100,size = (10,))

        
#     def optimize_rf(self):
#         self.rf_regressor = RandomizedSearchCV(self.rf_regressor, n_iter=10,
#                         param_distributions = {'max_depth': range(1, 15),
#                                                'min_samples_split': range(2, 50)},
#                         cv=5, n_jobs=-1, random_state=3,
#                         scoring='neg_mean_squared_error')
        

#     def test_1(self):
#         print(self.y)
#         self.rf_regressor.fit(self.x,self.y)
#         print(self.rf_regressor.predict(self.x))
#         print(self.rf_regressor.best_params_)
#         print(-self.rf_regressor.best_score_)

#     def test_2(self):
#         # for row in self.x:
#         #     print(self.rf_regressor.predict(row[None,:]))
#         #     # print(self.rf_regressor.apply(row[None,:]))
#         print(self.y_small)
#         print(self.rf_regressor.predict(self.x_small))

#     def test_3(self):
#         x_new = np.multiply(self.x,1)
#         y_new = np.multiply(self.y,1)
#         # x_new = np.copy(x_new)
#         self.rf_regressor.fit(x_new,y_new)
#         # print(self.rf_regressor.predict(x_new))
#         # print(self.rf_regressor.apply(x_new))
#         # print(y_new)
# test_class = test()
# test_class.optimize_rf()
# test_class.test_1()
# test_class.test_2()