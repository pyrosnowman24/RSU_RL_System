import numpy as np
import os, sys
from itertools import combinations

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent
from Models.RSU_Placement_Pointer.RSU_Intersection_Datamodule import RSU_Intersection_Datamodule

class Greedy_RSU():
    def __init__(self, simulation_agent):
        self.simulation_agent = simulation_agent

    def __call__(self, intersections, budget):
        normalized_intersections = self.normalize_items(intersections=intersections, budget=budget)
        sorted_indicies = np.argsort(normalized_intersections[:,-2],axis=0)
        sorted_intersections = normalized_intersections[sorted_indicies]
        cost = 0
        count = intersections.shape[0]
        while cost < budget and count > 0:
            count -= 1
            cost += sorted_intersections[count,-1]
        count += 1
        best_n_intersections = sorted_intersections[count:]
        reward = np.sum(best_n_intersections[:,-2])
        weight = np.sum(best_n_intersections[:,-1])
        return reward, weight, sorted_indicies[count:]
    
    def normalize_items(self,intersections,budget):
        intersections[:,-2] = np.divide(intersections[:,-2],np.max(intersections[:,-2]))
        intersections[:,-1] = np.divide(intersections[:,-1],budget)
        return intersections


if __name__ == '__main__':
    simulation_agent = Agent()
    budget = 5
    datamodule = RSU_Intersection_Datamodule(simulation_agent, n_scenarios=100, min_budget=15, max_budget=20, min_intersections = 40, max_intersections = 50)
    intersections, intersection_ids, budget = next(iter(datamodule.database))
    test = Greedy_RSU(simulation_agent)
    reward, best_pack = test(intersections = intersections, budget = budget)
    print(reward, best_pack)