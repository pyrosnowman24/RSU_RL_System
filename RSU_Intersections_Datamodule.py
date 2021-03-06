import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import pytorch_lightning as pl

class RSU_Intersection_Dataset(Dataset):
    def __init__(self,
                 agent,
                 n_scenarios: int = 100,
                 min_intersections: int = 10,
                 max_intersections: int = 15,
                 min_pre_rsu_network: int = 0,
                 max_pre_rsu_network: int = 5):
        self.n_scenarios = n_scenarios
        self.min_intersections = min_intersections
        self.max_intersections = max_intersections
        self.min_pre_rsu_network = min_pre_rsu_network
        self.max_pre_rsu_network = max_pre_rsu_network
        self.agent = agent

        self.sim_idx_array = []
        self.rsu_idx_array = []
        self.n_intersections = np.random.randint(low=min_intersections, high=max_intersections + 1, size=n_scenarios)
        self.n_pre_rsu_network = np.random.randint(low=min_pre_rsu_network, high=max_pre_rsu_network + 1, size=n_scenarios)
        for i,c in enumerate(self.n_intersections):
            self.sim_idx_array.append(np.random.choice(self.agent.network_intersections.shape[0],size = c,replace=False))
            self.rsu_idx_array.append(np.random.choice(self.sim_idx_array[-1].shape[0],size = self.n_pre_rsu_network[i],replace=False))

    def __len__(self):
        return self.n_scenarios

    def __getitem__(self,idx:int):
        intersections = self.agent.get_simulated_intersections(self.sim_idx_array[idx])
        intersection_idx = self.sim_idx_array[idx]
        rsu_network_idx = self.rsu_idx_array[idx]

        intersections_padded,intersection_idx_padded,rsu_network_idx_padded,mask_intersections, mask_rsu = self.pad_item(intersections, intersection_idx,rsu_network_idx)

        return intersections_padded, intersection_idx_padded, rsu_network_idx_padded, mask_intersections, mask_rsu

    def pad_item(self,
                 intersections: list,
                 intersections_idx: list,
                 rsu_network_idx: list):


        intersections_padded = np.zeros((self.max_intersections,intersections.shape[1]),dtype=np.float32)
        intersection_idx_padded = np.zeros((self.max_intersections),dtype=np.int64)
        rsu_network_idx_padded = np.zeros((self.max_pre_rsu_network),dtype=np.int64)

        mask_intersections = np.zeros((self.max_intersections),dtype=np.uint8)
        mask_rsu = np.zeros((self.max_pre_rsu_network),dtype=np.uint8)


        intersections_padded[:intersections.shape[0],:intersections.shape[1]] = intersections
        intersection_idx_padded[:len(intersections_idx)] = intersections_idx
        rsu_network_idx_padded[:len(rsu_network_idx)] = rsu_network_idx

        mask_intersections[:len(intersections_idx)] = 1
        mask_rsu[:len(rsu_network_idx)] = 1

        return intersections_padded,intersection_idx_padded,rsu_network_idx_padded,mask_intersections, mask_rsu

class RSU_Intersection_Datamodule(pl.LightningDataModule):
    def __init__(self,
                 agent,
                 batch_size: int = 16,
                 train_test_split: float = .7,
                 n_scenarios: int = 10,
                 min_intersections: int = 10,
                 max_intersections: int = 15,
                 min_pre_rsu_network: int = 0,
                 max_pre_rsu_network: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.n_scenarios = n_scenarios
        self.min_intersections = min_intersections
        self.max_intersections = max_intersections
        self.min_pre_rsu_network = min_pre_rsu_network
        self.max_pre_rsu_network = max_pre_rsu_network
        self.agent = agent
        self.setup()

    def setup(self):
        self.database = RSU_Intersection_Dataset(self.agent,
                                                 self.n_scenarios,
                                                 self.min_intersections,
                                                 self.max_intersections,
                                                 self.min_pre_rsu_network,
                                                 self.max_pre_rsu_network)
        train_size = int(self.train_test_split*len(self.database))
        test_size = len(self.database) - train_size
        self.train_dataset,self.test_dataset = random_split(self.database,[train_size,test_size])
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,num_workers=4)

