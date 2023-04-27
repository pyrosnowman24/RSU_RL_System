import numpy as np
import torch
import os,sys
from itertools import combinations
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(parentdir2)

from Models.Agent import Agent
from Models.RSU_Placement_Pointer.RSU_Intersection_Datamodule import RSU_Intersection_Datamodule
from Models.Knapsack_Algorithm.KA_RSU import KA_RSU
from Models.GA.GA_RSU import GA_RSU
from Models.Greedy.Greedy_RSU import Greedy_RSU
from Models.RSU_Placement_Pointer.RSU_RL_System import RSU_Placement_System

def calculate_best_reward_brute_force(intersections, budget):
        indices = np.arange(intersections.shape[1])
        best_reward = 0
        best_pack = None
        for i in range(1,budget+1):
            path_combinations = []
            path_combinations.extend(list(combinations(indices, i)))
            reward = np.empty(shape=(len(path_combinations)))
            reward = intersections[0,path_combinations,0]
            reward = reward.sum(axis=1)
            best_reward_i = reward.max()
            best_reward_index = reward.argmax()
            if best_reward_i > best_reward:
                best_reward = best_reward_i.clone()
                best_pack = path_combinations[best_reward_index]
        best_pack = np.asarray(best_pack)
        # best_items = intersections[0,best_pack,:]
        # diffs = best_items[:,0] - best_items[:,1]
        # diffs, indexes = torch.sort(diffs, descending=True)
        # best_pack = best_pack[indexes]
        return best_reward, np.sum(best_pack[:,-1]), best_pack

if __name__ == '__main__':
    simulation_agent = Agent()
    datamodule = RSU_Intersection_Datamodule(simulation_agent, n_scenarios=100, min_budget=5, max_budget=6, min_intersections = 40, max_intersections = 50, min_weight=0, max_weight=5)

    knapsack_algorithm = KA_RSU()

    genetic_algorithm = GA_RSU(population_size = 50, mutation_prob = .1)

    greedy_algorithm = Greedy_RSU(simulation_agent)

    directory_path = "/home/demo/RSU_RL_Placement/trained_models/"
    model_name = "dynamic_weights_3000_epochs_reduced_size_dropout"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_directory = os.path.join(directory_path,model_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,model_name)

    model = RSU_Placement_System(model_directory=model_directory)
    model.load_state_dict(torch.load(checkpoint_path))

    intersection, intersection_ids, budget = next(iter(datamodule.database))

    start = time.time()

    # BF_reward, BF_weight, BF_pack = calculate_best_reward_brute_force(intersections=intersection, budget=budget)
    # print("Genetic Algorithm:\n",BF_reward, BF_weight, BF_pack)

    GA_reward, GA_weight, GA_pack = genetic_algorithm(intersections=torch.tensor(intersection), epochs=100, budget=budget)
    print("Genetic Algorithm:\n",GA_reward, GA_weight, GA_pack)
    print(f"Runtime: {time.time()-start}")
    print('\n')
    start = time.time()

    KA_reward, KA_weight, KA_pack = knapsack_algorithm(intersections=torch.tensor(intersection[None,:,:]), budget=budget)
    print("Knapsack Algorithm:\n",KA_reward, KA_weight, KA_pack)
    print(f"Runtime: {time.time()-start}")
    print('\n')
    start = time.time()

    # log_pointer_scores, kp_pack, reward, weight = model(intersection[:,-2:], budget)
    # print("Pointer Network:\n",reward, weight, kp_pack)
    # print(f"Runtime: {time.time()-start}")
    # print('\n')
    # start = time.time()

    # greedy_reward, greedy_weight, greedy_pack = greedy_algorithm(intersections=intersection, budget=budget)
    # print("Greedy Algorithm:\n",greedy_reward, greedy_weight, greedy_pack)
    # print(f"Runtime: {time.time()-start}")
    # print('\n')
    # start = time.time()

    print(budget)
