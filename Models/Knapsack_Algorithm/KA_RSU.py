import numpy as np
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent
from Models.RSU_Placement_Pointer.RSU_Intersection_Datamodule import RSU_Intersection_Datamodule


class KA_RSU():
    def __init__(self, simulation_agent):
        self.simulation_agent = simulation_agent
        print(self.simulation_agent)
        self.init_intersections()

    def __call__(self, budget):
        K = [[0 for x in range(budget + 1)] for x in range(self.intersections.shape[0] + 1)]
        for i in range(self.intersections.shape[0] + 1):
            for w in range(budget + 1):
                if i == 0  or  w == 0:
                    K[i][w] = 0
                elif self.intersections[i-1,-1] <= w:
                    K[i][w] = max(self.intersections[i-1,-2]
                            + K[i-1][int(w-self.intersections[i-1,-1])],
                                K[i-1][w])
                else:
                    K[i][w] = K[i-1][w]
        return K[self.intersections.shape[0]][budget]
    
    def init_intersections(self):
        n_scenarios = 1
        max_budget = 5
        min_intersections = 12 # These two are the same because we dont want a random sized population
        max_intersections = 12
        datamodule = RSU_Intersection_Datamodule(self.simulation_agent, 
                                                 n_scenarios=n_scenarios, 
                                                 max_budget=max_budget, 
                                                 min_intersections=min_intersections, 
                                                 max_intersections=max_intersections)
        self.intersections = datamodule.database.scenarios[0,:,:]

simulation_agent = Agent()
test_class = KA_RSU(simulation_agent)
print(test_class(5))