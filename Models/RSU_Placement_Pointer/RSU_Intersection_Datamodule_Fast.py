import os,sys
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent


class RSU_Intersection_Dataset(Dataset):
    def __init__(self,
                 agent,
                 algorithm,
                 n_scenarios: int = 100,
                 min_intersections: int = 10,
                 max_intersections: int = 15,
                 min_budget: int = 2,
                 max_budget: int = 10,
                 min_weight: int = 0,
                 max_weight: int = 1,
                 ):
        self.agent = agent
        self.n_scenarios = n_scenarios
        self.min_intersections = min_intersections
        self.max_intersections = max_intersections
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.best_function_algorithm = algorithm

        self.sim_idx_array = []
        self.n_intersections = np.random.randint(low=min_intersections, high=max_intersections + 1, size=n_scenarios)
        self.budgets = np.arange(self.min_budget, self.max_budget+1, 1)
        self.create_scenarios()
        

    def __len__(self):
        return self.n_scenarios

    def __getitem__(self,idx:int):
        scenario_idx = self.scenario_idx[idx,:,:]
        scenario_idx = scenario_idx[scenario_idx>-1]
        scenario = self.scenarios[idx,:len(scenario_idx),:]
        budget = self.budgets[idx%self.budgets.shape[0]]
        best_pack = self.scenarios_best_packs[idx]
        best_pack = best_pack[best_pack>-1]
        best_reward = self.scenarios_best_rewards[idx]
        return scenario, scenario_idx, budget, best_pack, best_reward
    
    def create_scenarios(self):
        self.scenarios = np.ones(shape=(self.n_scenarios*self.budgets.shape[0], self.max_intersections, 7))
        self.scenarios_best_packs = -np.ones(shape=(self.scenarios.shape[0], self.max_intersections))
        self.scenarios_best_rewards = np.ones(shape=(self.scenarios.shape[0]))
        self.scenario_idx = -np.ones(shape=(self.scenarios.shape[0],self.max_intersections,1))
        index = 0
        for i in range(self.n_scenarios):
            for j in range(self.budgets.shape[0]):
                self.scenario_idx[index,:self.n_intersections[i]] = np.random.choice(self.agent.network_intersections.shape[0],size = (self.n_intersections[i],1) , replace=False)
                for k in range(self.n_intersections[i]):
                    self.scenarios[index,k,:-1] = self.agent.get_simulated_intersections(self.scenario_idx[index,k])
                    self.scenarios[index,k,-1] = np.random.uniform(self.min_weight,self.max_weight)
                best_reward, best_weight, best_pack = self.calculate_best_reward(self.scenarios[index, :len(self.scenario_idx[index,self.scenario_idx[index]>-1])], self.budgets[j])
                self.scenarios_best_packs[index,:best_pack.shape[0]] = best_pack
                self.scenarios_best_rewards[index] = best_reward
                index += 1

    def calculate_best_reward(self,intersections, budget):
        return self.best_function_algorithm(torch.tensor(intersections), torch.tensor(budget))

class RSU_Intersection_Datamodule(pl.LightningDataModule):
    def __init__(self,
                 agent,
                 algorithm,
                 batch_size: int = 1,
                 train_test_split: float = .6,
                 n_scenarios: int = 100,
                 min_intersections: int = 10,
                 max_intersections: int = 15,
                 min_budget: int = 2,
                 max_budget: int = 10,
                 min_weight: int = 0,
                max_weight: int = 1,
                 ):
        super().__init__()
        self.agent = agent
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.n_scenarios = n_scenarios
        self.min_intersections = min_intersections
        self.max_intersections = max_intersections
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.setup()

    def setup(self, stage: str = None):
        self.database = RSU_Intersection_Dataset(
                                    self.agent,
                                    self.algorithm,
                                    self.n_scenarios,
                                    self.min_intersections,
                                    self.max_intersections,
                                    self.min_budget,
                                    self.max_budget,
                                    self.min_weight,
                                    self.max_weight,
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

# simulation_agent = Agent()
# knapsack_algorithm = KA_RSU()
# datamodule = RSU_Intersection_Datamodule(simulation_agent, knapsack_algorithm, n_scenarios=10, min_budget=2, max_budget=4)
# print(next(iter(datamodule.database)))