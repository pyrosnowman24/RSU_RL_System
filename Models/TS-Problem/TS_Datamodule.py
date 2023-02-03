import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler


class TP_Dataset(Dataset):
    def __init__(self,
                 n_scenarios: int = 100,
                 n_cities: int = 10):
        self.n_scenarios = n_scenarios
        self.n_cities = n_cities

        self.sim_idx_array = []
        self.scenarios = np.random.rand(self.n_scenarios, self.n_cities, 2)

    def __len__(self):
        return self.n_scenarios

    def __getitem__(self,idx:int):
        scenario = self.scenarios[idx]

        return scenario
    
    # def __getitem__(self,idx:int):
    #     # This version creates a new scenario each time its called
    #     intersection_idx = np.random.choice(self.agent.network_intersections.shape[0],size = np.random.randint(low=self.min_intersections, high=self.max_intersections + 1) , replace=False)
    #     rsu_network_idx = np.random.choice(intersection_idx.shape[0],size = np.random.randint(low=self.min_pre_rsu_network, high=self.max_pre_rsu_network + 1),replace=False)

    #     intersections = self.agent.get_simulated_intersections(intersection_idx)
    #     # scaled_intersections = self.scale_intersections(intersections)
    #     intersections_padded,intersection_idx_padded,rsu_network_idx_padded,mask = self.pad_item(intersections, intersection_idx,rsu_network_idx)

    #     return intersections_padded, intersection_idx_padded, rsu_network_idx_padded, mask

class TS_Datamodule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 1,
                 train_test_split: float = .6,
                 n_scenarios: int = 100,
                 n_cities: int = 10):
        super().__init__()
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.n_scenarios = n_scenarios
        self.n_cities = n_cities
        self.setup()

    def setup(self, stage: str = None):
        self.database = TP_Dataset(
                                    self.n_scenarios,
                                    self.n_cities
                                )
        train_size = int(self.train_test_split*len(self.database))
        test_size = len(self.database) - train_size
        self.train_dataset,self.test_dataset = random_split(self.database,[train_size,test_size])        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,num_workers=4)


# datamodule = TS_Datamodule()
# print(next(iter(datamodule.database)))