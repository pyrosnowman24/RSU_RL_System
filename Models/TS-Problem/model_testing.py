import torch
import pandas as pd
from torch import Tensor, nn
import numpy as np
import os
from pytorch_lightning import LightningModule, Trainer


from TS_RL_System import TS_Datamodule

from TS_RL_System import TS_System

if __name__ == '__main__':
    max_epochs = 10000

    directory_path = "/home/demo/RSU_RL_Placement/trained_models/"
    model_name = "ts_new_reward_5_cities_256_hidden_5000_epochs_ts_loss"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_directory = os.path.join(directory_path,model_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,model_name)

    model = TS_System(model_directory = model_directory)
    model.load_state_dict(torch.load(checkpoint_path))
    datamodule = TS_Datamodule(n_scenarios=100, n_cities=5)
    trainer = Trainer(max_epochs = max_epochs)

    trainer.test(model,datamodule=datamodule)
