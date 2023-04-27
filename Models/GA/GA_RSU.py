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
from Models.RSU_Placement_Pointer.RSU_Intersection_Datamodule import RSU_Intersection_Datamodule


class GA_RSU():
    def __init__(self,population_size,mutation_prob):
        self.population_size = population_size
        self.mutation_prob = mutation_prob

    def __call__(self, intersections, budget, epochs=100):
        self.epochs = epochs
        self.budget = budget
        normalized_intersections = self.normalize_items(intersections=intersections, budget=budget)
        population = self.init_population(normalized_intersections)
        best_individual = np.zeros(population[0].shape[0])
        best_score = 0
        best_weight = 0

        for i in range(self.epochs):
            # print("Epoch ",i," Best Score: ",best_score)
            scores, weights = self.scoring(population, normalized_intersections)
            best_individual_epoch, best_individual_weight, best_score_epoch = self.best_individual(population, scores, weights)
            if best_score_epoch > best_score:
                best_individual = best_individual_epoch
                best_score = best_score_epoch
                best_weight = best_individual_weight
            parents = self.selection(population,scores)
            new_population = self.combination(parents)
            population = self.mutation(new_population)
        best_individual_index = np.arange(normalized_intersections.shape[0])[best_individual.astype(bool)]
        best_score, best_weight = self.individual_score(best_individual, normalized_intersections)
        return best_score, best_weight, best_individual_index  

    def init_intersections(self):
        n_scenarios = 1
        max_budget = 5
        min_intersections = 12 # These two are the same because we dont want a random sized population
        max_intersections = 12
        datamodule = RSU_Intersection_Datamodule(self.agent, 
                                                 n_scenarios=n_scenarios, 
                                                 max_budget=max_budget, 
                                                 min_intersections=min_intersections, 
                                                 max_intersections=max_intersections)
        self.intersections = datamodule.database.scenarios[0,:,:]

    def init_population(self, intersections):
        population = np.zeros(shape = (self.population_size, intersections.shape[0]))
        pop_size = np.random.choice(np.arange(1,intersections.shape[0]-1,1), size = self.population_size)
        for i in range(population.shape[0]):
            pop_indices = np.random.choice(range(population.shape[1]),size = pop_size[i],replace=False)
            population[i,pop_indices] = 1
        return population

    def scoring(self,population, intersections):
        scores = np.zeros(shape = population.shape[0])
        weights = np.zeros(shape = population.shape[0])
        for i in range(scores.shape[0]):
            individual_intersections = intersections[population[i] == 1]
            reward = torch.sum(individual_intersections[:,-2])
            weight = torch.sum(individual_intersections[:,-1])
            if weight > self.budget:
                scores[i] = self.budget- weight
                weights[i] = weight
            else:
                scores[i] = reward
                weights[i] = weight
        return scores, weights

    def selection(self, population, scores):
        scores_sorted_idx = np.argsort(scores)
        halfway = int(len(scores)/2)
        parent_idx = scores_sorted_idx[-halfway:]
        parents = population[scores_sorted_idx[parent_idx]]
        return parents
    
    def combination(self, parents):
        new_individuals = np.zeros(shape=(parents.shape[0], parents.shape[1]))
        for i in range(new_individuals.shape[0]):
            pair = np.random.choice(range(new_individuals.shape[0]),size = 2, replace=False)
            crossover = np.random.randint(1,parents.shape[1]-1)
            first_half = parents[pair[0]][:crossover]
            second_half = parents[pair[1]][crossover:]
            new_individuals[i] = np.concatenate((first_half,second_half))
        new_population = np.vstack((parents,new_individuals))
        return new_population
    
    def mutation(self, new_population):
        for individual in new_population:
            mutation_var = np.random.rand(1)
            if mutation_var < self.mutation_prob:
                idx = np.random.choice(range(individual.shape[0]))
                individual[idx] = 1 - individual[idx]
        return new_population
    
    def best_individual(self,population, scores, weights):
        idx = np.argmax(scores)
        return population[idx], weights[idx], scores[idx]
    
    def rsu_network_from_individual(self,individual,intersections):
        return intersections[individual==1]
    
    def individual_score(self,individual, intersections):
        individual_intersections = intersections[individual == 1]
        reward = torch.sum(individual_intersections[:,-2])
        weight = torch.sum(individual_intersections[:,-1])
        if weight > self.budget:
            score = reward + (self.budget- weight)
        else:
            score = reward
        return score, weight
    
    def calculate_best_reward(self,intersections):
        indices = np.arange(intersections.shape[0])
        best_reward = 0
        best_pack = None
        for i in range(1,self.budget+1):
            path_combinations = []
            path_combinations.extend(list(combinations(indices, i)))
            reward = np.empty(shape=(len(path_combinations)))
            reward = intersections[path_combinations,-2]
            reward = reward.sum(axis=1)
            best_reward_i = reward.max()
            best_reward_index = reward.argmax()
            if best_reward_i > best_reward:
                best_reward = best_reward_i
                best_pack = path_combinations[best_reward_index]
        best_pack = np.asarray(best_pack)
        best_individual = np.zeros(intersections.shape[0])
        best_individual[best_pack] = 1
        return best_reward, best_individual
    
    def normalize_items(self,intersections,budget):
        intersections[:,-2] = torch.divide(intersections[:,-2],torch.max(intersections[:,-2]))
        intersections[:,-1] = torch.divide(intersections[:,-1],budget)
        return intersections
                    
# if __name__ == '__main__':
#     simulation_agent = Agent()
#     test = GA_RSU(simulation_agent, population_size = 50, mutation_prob = .1)
#     test(epochs = 300, budget = 10)
