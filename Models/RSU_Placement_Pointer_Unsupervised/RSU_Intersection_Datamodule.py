import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from Models.Knapsack_Algorithm.KA_RSU import KA_RSU



class RSU_Intersection_Dataset(Dataset):
    def __init__(self,
                 agent,
                 n_scenarios: int = 100,
                 min_intersections: int = 10,
                 max_intersections: int = 15,
                 min_budget: int = 2,
                 max_budget: int = 10
                 ):
        self.agent = agent
        self.n_scenarios = n_scenarios
        self.min_intersections = min_intersections
        self.max_intersections = max_intersections
        self.min_budget = min_budget
        self.max_budget = max_budget

        self.sim_idx_array = []
        self.n_intersections = np.random.randint(low=min_intersections, high=max_intersections + 1, size=n_scenarios)
        self.budgets = np.random.randint(self.min_budget,self.max_budget, size = self.n_scenarios)
        self.knapsack_algorithm = KA_RSU()
        self.create_scenarios()
        

    def __len__(self):
        return self.n_scenarios

    def __getitem__(self,idx:int):
        scenario_idx = self.scenario_idx[idx,:,:]
        scenario_idx = scenario_idx[scenario_idx>-1]

        scenario = self.scenarios[idx,:len(scenario_idx),:]

        budget = self.budgets[idx]
        return scenario, scenario_idx, budget
    
    def create_scenarios(self):
        self.scenarios = np.ones(shape=(self.n_scenarios, self.max_intersections, 7))
        self.scenario_idx = -np.ones(shape=(self.n_scenarios,self.max_intersections,1))
        for i in range(self.n_scenarios):
            self.scenario_idx[i,:self.n_intersections[i]] = np.random.choice(self.agent.network_intersections.shape[0],size = (self.n_intersections[i],1) , replace=False)
            for j in range(self.n_intersections[i]):
                self.scenarios[i,j,:-1] = self.agent.get_simulated_intersections(self.scenario_idx[i,j])
                self.scenarios[i,j,-1] = np.random.uniform(0,1)

    def calculate_best_reward_knapsack(self,intersections):
        return self.knapsack_algorithm(intersections, self.budget)

class RSU_Intersection_Datamodule(pl.LightningDataModule):
    def __init__(self,
                 agent,
                 batch_size: int = 1,
                 train_test_split: float = .6,
                 n_scenarios: int = 100,
                 min_intersections: int = 10,
                 max_intersections: int = 15,
                 min_budget: int = 2,
                 max_budget: int = 10
                 ):
        super().__init__()
        self.agent = agent
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.n_scenarios = n_scenarios
        self.min_intersections = min_intersections
        self.max_intersections = max_intersections
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.setup()

    def setup(self, stage: str = None):
        self.database = RSU_Intersection_Dataset(
                                    self.agent,
                                    self.n_scenarios,
                                    self.min_intersections,
                                    self.max_intersections,
                                    self.min_budget,
                                    self.max_budget
                                )
        train_size = int(self.train_test_split*len(self.database))
        test_size = len(self.database) - train_size
        self.train_dataset,self.test_dataset = random_split(self.database,[train_size,test_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,num_workers=4)
