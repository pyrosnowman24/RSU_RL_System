import numpy as np


class RewardFunctions():
    def __call__(self, model, features, rsu_network, network_intersections, W, simulation_time):
        self.features = features
        self.network_intersections = network_intersections[:,1:-1]
        self.rsu_network = rsu_network[:,:]
        self.W = W
        self.simulation_time = simulation_time

        if model == "Policy Gradient":
            reward = self.reward_pg()
        if model == "Q Learning":
            reward = self.reward_ql()
        if model == "Q Learning Positive":
            rewards, reward, features = self.reward_positive_ql()
        else:
            reward = self.reward()
        return rewards, reward, features

    def reward(self):
        """Reward for Actor Critic based model. The higher the reward the better the solution.

        Args:
            features (numpy.array): The valeus of each feature for all RSUs in network.
            W (list): Used to bias rewards between features

        Returns:
            int: Reward for all RSUs in RSU network
        """
        print(self.features)
        avg_features = np.nanmean(self.features,axis=1)
        avg_features[0] = -(300/avg_features[0])
        avg_features[1] = 200/avg_features[1,:]
        print(avg_features)
        reward = np.multiply(avg_features,W)
        reward = np.sum(reward)
        reward = reward - len(reward)
        return reward

    def reward_pg(self):
        """Reward for Policy Gradient based model. The lower the reward the better the solution.

        Args:
            features (numpy.array): The valeus of each feature for all RSUs in network.
            W (list): Used to bias rewards between features

        Returns:
            int: Reward for all RSUs in RSU network
        """
        # print("features",features)
        features = self.features
        features[0] = np.nan_to_num(features[0],nan = -104)
        features[1] = np.nan_to_num(features[1],nan = 0)
        # features[0] = np.power(100/(features[0,:]+30),2)
        # features[1] = .025*features[1]
        features[0] = np.power(50/(features[0,:]+105),2)
        features[1] = 500/(features[1,:]+1)

        # print("processed features",features)

        features = np.sum(features,axis=0)
        for i in range(len(features)):
            features[i] += .15 * np.square(i)
        # print(features)
        return features

    def reward_ql(self):
        """Reward for Q-Learning based model. The higher the reward the better the solution.

        Args:
            features (numpy.array): The valeus of each feature for all RSUs in network.
            W (list): Used to bias rewards between features

        Returns:
            int: Reward for all RSUs in RSU network
        """
        # print("features",features)
        print(features)
        features = self.features
        features[0] = np.nan_to_num(features[0],nan = -104)
        features[1] = np.nan_to_num(features[1],nan = 0)
        features[0] = .005 * np.power(features[0,:]+104,2)
        features[1] = .00005 * np.power(features[1,:],2)
        # print(features)

        # print("processed features",features)

        features = np.sum(features,axis=0)
        for i in range(len(features)):
            features[i] -= .10 * np.square(i)
        # print(features)
        return features

    def reward_positive_ql(self):
        """Reward for Q-Learning based model. The higher the reward the better the solution. This version will always be positive.

        Args:
            features (numpy.array): The valeus of each feature for all RSUs in network.
            W (list): Used to bias rewards between features

        Returns:
            int: Reward for all RSUs in RSU network
        """
        rewards = np.zeros((4,self.features.shape[1]))
        rewards[0,:] = self.received_dcb_reward(self.features[0,:])
        rewards[1,:] = self.number_received_msg_reward(self.features[1,:])
        rewards[2,:] = self.degree_centrality_reward()
        rewards[3,:] = self.rsu_network[:,-1]
        
        rewards_sum = np.sum(rewards,axis=0)
        return rewards, rewards_sum, self.features

    def received_dcb_reward(self,data):
        data = np.nan_to_num(data, nan = -104)
        data = np.clip(data,-104,-50) # Clips decibels within expected range.
        data = .25*np.sin(((data+77)*np.pi)/54)+.25
        print(data)
        return data
        
    def number_received_msg_reward(self,data):
        data = np.divide(data,self.simulation_time[1]-self.simulation_time[0])
        data = np.clip(data,0,500) # Clips number of messages/time step.
        data = .25*np.sin(((data-250)*np.pi)/500)+.25
        print(data)
        return data

    def degree_centrality_reward(self):
        data = self.rsu_network[:,-2]*.1
        return data
