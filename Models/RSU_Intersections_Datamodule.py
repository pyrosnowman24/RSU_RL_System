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

        # self.sim_idx_array = []
        # self.rsu_idx_array = []
        # self.n_intersections = np.random.randint(low=min_intersections, high=max_intersections + 1, size=n_scenarios)
        # self.n_pre_rsu_network = np.random.randint(low=min_pre_rsu_network, high=max_pre_rsu_network + 1, size=n_scenarios)
        # for i,c in enumerate(self.n_intersections):
        #     self.sim_idx_array.append(np.random.choice(self.agent.network_intersections.shape[0],size = c,replace=False))
        #     self.rsu_idx_array.append(np.random.choice(self.sim_idx_array[-1].shape[0],size = self.n_pre_rsu_network[i],replace=False))

    def __len__(self):
        return self.n_scenarios

    # def __getitem__(self,idx:int):
    #     intersections = self.agent.get_simulated_intersections(self.sim_idx_array[idx])
    #     intersection_idx = self.sim_idx_array[idx]
    #     rsu_network_idx = self.rsu_idx_array[idx]

    #     intersections_padded,intersection_idx_padded,rsu_network_idx_padded,mask = self.pad_item(intersections, intersection_idx,rsu_network_idx)

    #     return intersections_padded, intersection_idx_padded, rsu_network_idx_padded, mask
    
    def __getitem__(self,idx:int):
        # This version creates a new scenario each time its called
        intersection_idx = np.random.choice(self.agent.network_intersections.shape[0],size = np.random.randint(low=self.min_intersections, high=self.max_intersections + 1) , replace=False)
        rsu_network_idx = np.random.choice(intersection_idx.shape[0],size = np.random.randint(low=self.min_pre_rsu_network, high=self.max_pre_rsu_network + 1),replace=False)
        rsu_network_idx

        intersections = self.agent.get_simulated_intersections(intersection_idx)

        intersections_padded,intersection_idx_padded,rsu_network_idx_padded,mask = self.pad_item(intersections, intersection_idx,rsu_network_idx)

        return intersections_padded, intersection_idx_padded, rsu_network_idx_padded, mask

    def pad_item(self,
                 intersections: list,
                 intersections_idx: list,
                 rsu_network_idx: list):
        """Responsible for padding the subset of intersections to the maximum number possible for the problem. Also adds an extra "intersections" at the beginning, which the model can select to not place an RSU.

        Args:
            intersections (list): A subset of intersections for the batch.
            intersections_idx (list): indices of each intersection in the batch from the complete list of intersections.
            rsu_network_idx (list): The index of the intersections that will have an RSU at the start of an iteration.

        Returns:
            intersections_padded (list): List of intersections with extra added to the start and padding at the end.
            intersection_idx_padded (list): List of intersection indices with extra added to the front and padding to the end.
            rsu_network_idx_padded (list): List of indices for RSUs with padding added to the end for the maximum RSU network size for pre-placed RSUs.
            mask (list): Mask that can be applied to the intersections_padded to remove padding. This does not remove the extra value at the beginning.
        """


        intersections_padded = np.zeros((self.max_intersections+1,intersections.shape[1]),dtype=np.float32)
        intersection_idx_padded = np.zeros((self.max_intersections+1),dtype=np.int64)
        rsu_network_idx_padded = np.zeros((self.max_pre_rsu_network),dtype=np.int64)

        intersection_mask = np.zeros((self.max_intersections+1),dtype=np.uint8)
        rsu_mask = np.zeros((self.max_pre_rsu_network),dtype=np.uint8)

        intersections_padded[1:intersections.shape[0]+1,:intersections.shape[1]] = intersections
        intersection_idx_padded[1:len(intersections_idx)+1] = intersections_idx
        rsu_network_idx_padded[:len(rsu_network_idx)] = rsu_network_idx

        intersection_mask[:len(intersections_idx)+1] = 1
        rsu_mask[:len(rsu_network_idx)] = 1

        intersection_mask = np.where(intersection_mask == 1, True, False)
        rsu_mask = np.where(rsu_mask == 1, True, False)

        mask = self.combine_masks(intersection_mask,rsu_network_idx,rsu_mask)

        return intersections_padded,intersection_idx_padded,rsu_network_idx_padded,mask

    def combine_masks(self,intersection_mask,rsu_network_idx,rsu_mask):
        """Combines the masks for the padding and the already selected intersections.

        Args:
            intersection_mask (numpy.ndarray): Mask for padding in intersections
            rsu_network_idx (numpy.ndarray): IDs of intersections that were pre-selected for the RSU network
            rsu_mask (numpy.ndarray): Mask for padding in the RSU network IDs

        Returns:
            numpy.ndarray: Mask that removes both the padding and already selected RSUs.
        """
        mask = intersection_mask.copy()
        for i in range(rsu_mask.shape[0]):
            if bool(rsu_mask[i]) is True:
                mask[rsu_network_idx[i]+1] = False
        return mask

class RSU_Intersection_Datamodule(pl.LightningDataModule):
    def __init__(self,
                 agent,
                 batch_size: int = 1,
                 train_test_split: float = 1,
                 n_scenarios: int = 100,
                 min_intersections: int = 10,
                 max_intersections: int = 30,
                 min_pre_rsu_network: int = 0,
                 max_pre_rsu_network: int = 4):
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

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size = self.batch_size,num_workers=4)

