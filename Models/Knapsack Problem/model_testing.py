import torch
import pandas as pd
from torch import Tensor, nn
import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer


from TS_RL_System import TS_Datamodule

from TS_RL_System import TS_System

def test_model(model,datamodule, trainer):
    trainer.test(model,datamodule=datamodule)

def plot_solution(model,dataloader):
    data = next(iter(dataloader))
    ts_path, _ = model(data)
    best_reward, best_path = model.calculate_best_reward(data)
    model.best_reward = best_reward
    ts_reward, loss = model.ts_loss(data, ts_path[0,:])

    print("best path = ",best_path)
    print("ts_path = ",ts_path)

    print("best reward = ",best_reward)
    print("ts_path reward = ",ts_reward)
    print("loss = ",loss)

    best_path_cities = data[0][best_path,:]
    ts_path_cities = data[0][ts_path,:][0]

    print(best_path_cities)

    print(ts_path_cities)

    fig,(ax1,ax2) = plt.subplots(2)
    ax1.scatter(data[0,:,0],data[0,:,1])
    ax2.scatter(data[0,:,0],data[0,:,1])
    ax1.plot(best_path_cities[:,0],best_path_cities[:,1], 'r')
    ax2.plot(ts_path_cities[:,0],ts_path_cities[:,1], 'r')
    plt.show()

def solution_stats(model,dataloader):
    losses = []
    it = iter(dataloader)
    for i in range(600):
        data = next(it)
        ts_path, _ = model(data)
        best_reward, best_path = model.calculate_best_reward(data)
        model.best_reward = best_reward
        ts_reward, loss = model.ts_loss(data, ts_path[0,:])
        losses.append(loss)

    fig,ax = plt.subplots(1)
    ax.hist(losses)
    plt.show()
if __name__ == '__main__':
    max_epochs = 10000

    directory_path = "/home/demo/RSU_RL_Placement/trained_models/"
    model_name = "ts_new_reward_7_cities_256_hidden_5000_epochs_ts_loss"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_directory = os.path.join(directory_path,model_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,model_name)

    model = TS_System(model_directory = model_directory)
    model.load_state_dict(torch.load(checkpoint_path))
    datamodule = TS_Datamodule(n_scenarios=1000, n_cities=7)
    trainer = Trainer(max_epochs = max_epochs)

    solution_stats(model,datamodule.train_dataloader())