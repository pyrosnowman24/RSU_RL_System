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


class GA_RSU():
    def __init__(self,agent,population_size,mutation_prob):
        self.agent = agent
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.init_intersections()

    def __call__(self,epochs, budget):
        self.epochs = epochs
        self.budget = budget
        population = self.init_population()
        best_individual = None
        best_score = 0

        for i in range(self.epochs):
            print("Epoch ",i," Best Score: ",best_score)
            scores = self.scoring(population)
            best_individual_epoch, best_score_epoch = self.best_individual(population, scores)
            if best_score_epoch > best_score:
                best_individual = best_individual_epoch
                best_score = best_score_epoch
            parents = self.selection(population,scores)
            new_population = self.combination(parents)
            population = self.mutation(new_population)
        print(best_individual)
        print(self.individual_score(best_individual))     
        optimal_score, optimal_individual = self.calculate_best_reward(self.intersections)
        print(optimal_score)
        print(self.individual_score(optimal_individual))

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

    def init_population(self):
        population = np.random.randint(2, size = (self.population_size, self.intersections.shape[0]))
        return population

    def scoring(self,population):
        scores = np.zeros(shape = population.shape[0])
        for i in range(scores.shape[0]):
            individual_intersections = self.intersections[population[i] == 1]
            reward = np.sum(individual_intersections[:,-2])
            weight = np.sum(individual_intersections[:,-1])
            if weight > self.budget:
                scores[i] = self.budget- weight
            else:
                scores[i] = reward
        return scores

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
    
    def best_individual(self,population, scores):
        idx = np.argmax(scores)
        return population[idx], scores[idx]
    
    def rsu_network_from_idnividual(self,individual):
        return self.intersections[individual==1]
    
    def individual_score(self,individual):
        individual_intersections = self.intersections[individual == 1]
        reward = np.sum(individual_intersections[:,-2])
        weight = np.sum(individual_intersections[:,-1])
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
        print(best_pack)
        print(best_individual)
        return best_reward, best_individual
                    
if __name__ == '__main__':
    simulation_agent = Agent()
    test = GA_RSU(simulation_agent, 50, .1)
    test(300, 10)
