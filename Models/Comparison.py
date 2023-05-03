import numpy as np
import torch
import os,sys
from itertools import combinations
import time
import matplotlib.pyplot as plt

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
    # datamodule = RSU_Intersection_Datamodule(simulation_agent, n_scenarios=100, min_budget=5, max_budget=6, min_intersections = 40, max_intersections = 50, min_weight=0, max_weight=5)
    datamodule = RSU_Intersection_Datamodule(simulation_agent, n_scenarios=1000, min_budget=9, max_budget=10, min_intersections = 10, max_intersections = 25, min_weight=1, max_weight=3)

    knapsack_algorithm = KA_RSU()

    genetic_algorithm = GA_RSU(population_size = 50, mutation_prob = .1)

    greedy_algorithm = Greedy_RSU(simulation_agent)

    directory_path = "/home/demo/RSU_RL_Placement/trained_models/"
    model_name = "knapsack_300_epochs_100_scenarios_final3"
    model_directory = os.path.join(directory_path,model_name+'/')
    model_path = os.path.join(model_directory,model_name)

    checkpoint_directory = os.path.join(directory_path,model_name+'/')
    checkpoint_path = os.path.join(checkpoint_directory,model_name)

    model = RSU_Placement_System(model_directory=model_directory)
    model.load_state_dict(torch.load(checkpoint_path))
    GA_avg_reward = []
    GA_avg_weight = []
    GA_avg_runtime = []

    KA_avg_reward = []
    KA_avg_weight = []
    KA_avg_runtime = []

    pointer_avg_reward = []
    pointer_avg_weight = []
    pointer_avg_runtime = []

    greedy_avg_reward = []
    greedy_avg_weight = []
    greedy_avg_runtime = []

    start = time.time()

    for i in range(500):

        intersection, intersection_ids, budget = datamodule.database.__getitem__(i)
        # print(intersection[:,-1])

        # GA_reward, GA_weight, GA_pack = genetic_algorithm(intersections=torch.tensor(intersection), epochs=100, budget=budget)
        # # print("Genetic Algorithm:\n",GA_reward, GA_weight, GA_pack)
        # # print(f"Runtime: {time.time()-start}")
        # # print('\n')
        # GA_avg_reward.append(GA_reward)
        # GA_avg_weight.append(budget - GA_weight)
        # GA_avg_runtime.append(time.time()-start)
        # start = time.time()

        # KA_reward, KA_weight, KA_pack = knapsack_algorithm(intersections=torch.tensor(intersection), budget=budget)
        # # print("Knapsack Algorithm:\n",KA_reward, KA_weight, KA_pack)
        # # print(f"Runtime: {time.time()-start}")
        # # print('\n')
        # KA_avg_reward.append(KA_reward)
        # KA_avg_weight.append(budget - KA_weight)
        # KA_avg_runtime.append(time.time()-start)
        # start = time.time()

        log_pointer_scores, kp_pack, reward, weight = model(intersection[:,-2:], budget)
        # print("Pointer Network:\n",reward, weight, kp_pack)
        # print(f"Runtime: {time.time()-start}")
        # print('\n')
        pointer_avg_reward.append(reward)
        pointer_avg_weight.append(weight-budget)
        pointer_avg_runtime.append(time.time()-start)
        start = time.time()

        # greedy_reward, greedy_weight, greedy_pack = greedy_algorithm(intersections=intersection, budget=budget)
        # # print("Greedy Algorithm:\n",greedy_reward, greedy_weight, greedy_pack)
        # # print(f"Runtime: {time.time()-start}")
        # # print('\n')
        # greedy_avg_reward.append(greedy_reward)
        # greedy_avg_weight.append(budget - greedy_weight)
        # greedy_avg_runtime.append(time.time()-start)
        # start = time.time()

    # print(np.mean(GA_avg_reward), np.mean(GA_avg_weight), np.mean(GA_avg_runtime))
    # print(np.mean(KA_avg_reward), np.mean(KA_avg_weight), np.mean(KA_avg_runtime))
    # print(np.mean(pointer_avg_reward), np.mean(pointer_avg_weight), np.mean(pointer_avg_runtime))
    # print(np.mean(greedy_avg_reward), np.mean(greedy_avg_weight), np.mean(greedy_avg_runtime))
    # print(budget)

    fig, (ax1,ax2,ax3) = plt.subplots(3)
    ax1.plot(pointer_avg_reward)
    ax2.plot(pointer_avg_weight)
    ax3.plot(pointer_avg_runtime)
    ax1.set_title("Reward, weight, and runtime of pointer network over 500 scenarios")
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Reward")
    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Estimated Budget - Budget")
    ax3.set_xlabel("Scenario")
    ax3.set_ylabel("Runtime")
    plt.show()

