import numpy as np
import torch
import os, sys
from itertools import combinations


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent

class KA_RSU():
    def __init__(self):
        self.howdy = 1

    def __call__(self, intersections, budget):
        intersections = self.normalize_items(intersections=intersections, budget=budget)
        i = 1
        last_reward = None
        last_weight = None
        last_pack = None
        while i < intersections.shape[0]:
            ka_budget = i
            reward, weight, pack = self.ka_algorithm(intersections, ka_budget)
            if weight < budget:
                last_reward = reward
                last_weight = weight
                last_pack = pack
                i += 1
            else:
                break
        return last_reward, last_weight, last_pack 
    
    def ka_algorithm(self, intersections, budget):
        if len(intersections.shape) > 2: intersections = intersections[0,:,:]
        self.budget = budget
        self.intersections = intersections
        self.K = [[0 for x in range(budget + 1)] for x in range(self.intersections.shape[0] + 1)]
        for i in range(self.intersections.shape[0] + 1):
            for w in range(budget + 1):
                if i == 0  or  w == 0:
                    self.K[i][w] = 0
                elif self.intersections[i-1,-1] <= w:
                    a = self.intersections[i-1,-2] + self.K[i-1][int(w-self.intersections[i-1,-1])]
                    b = self.K[i-1][w]
                    
                    if a > b: # Add new item
                        self.K[i][w] = a
                    else: # Ignore item
                        self.K[i][w] = b
                    # self.K[i][w] = max(self.intersections[i-1,-2]
                            # + self.K[i-1][int(w-self.intersections[i-1,-1])],
                            #     self.K[i-1][w])
                else:
                    self.K[i][w] = self.K[i-1][w]
        best_reward = self.K[self.intersections.shape[0]][budget]
        best_pack = self.reconstruct(self.intersections.shape[0],budget)
        best_pack_mask = np.zeros(shape = self.intersections.shape[0])
        best_pack_mask[best_pack.astype(int)] = 1
        best_reward_check = torch.sum(self.intersections[best_pack_mask.astype(bool)][:,-2], axis = 0)
        best_weights = torch.sum(self.intersections[best_pack_mask.astype(bool)][:,-1], axis = 0)
        return best_reward_check, best_weights, np.sort(best_pack).astype(int)


    def reconstruct(self, i, w):
        i = int(i)
        w = int(w)
        if i == 0: 
            # base case
            return None
        if self.K[i][w] > self.K[int(i-1)][w]:
            # we have to take item i
            recursive_value = self.reconstruct(i-1, w - self.intersections[i-1,-1])
            if np.any(recursive_value == None):
                array = np.empty(shape=(1,))
                array[0] = i-1
                return array
            else:
                array = np.empty(shape=(1,))
                array[0] = i-1
                return np.concatenate((array,recursive_value))
        else:
            # we don't need item i
            return self.reconstruct(i-1, w)
        
    def normalize_items(self,intersections,budget):
        intersections[:,-2] = torch.divide(intersections[:,-2],torch.max(intersections[:,-2]))
        intersections[:,-1] = torch.divide(intersections[:,-1],torch.tensor(budget))
        return intersections

# simulation_agent = Agent()
# test_class = KA_RSU()
# best_reward, best_pack_mask = test_class(intersections, 3)