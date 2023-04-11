from cmath import nan
from enum import unique
from operator import length_hint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re
from scipy import stats, special

directory_path = "/home/demo/RSU_RL_Placement/trained_models/"
model_name = "test2"
model_directory = os.path.join(directory_path,model_name+'/')
history_path = os.path.join(model_directory,"model_history.csv")

history_df = pd.read_csv(history_path, on_bad_lines='skip', engine = 'python')

ts_path_history = history_df['kp_pack']
reward_history = history_df['reward']
avg_rsu = 0

for kp_pack in ts_path_history:
    results = re.findall(r"\d+",kp_pack)
    avg_rsu += len(results)
    avg_rsu /= 2

loss = history_df['loss'].to_numpy()

def plot_single_loss():
    x = np.arange(0,len(loss)+1,step = 1000)
    average_loss = []
    for i in range(len(x)-1):
        average_loss.append(np.nanmean(loss[x[i]:x[i+1]]))
    print(average_loss)
    fig,ax = plt.subplots(1)
    ax.plot(loss)
    ax.plot(x[1:],average_loss)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Actor Loss")
    ax.set_title("Actor Loss over 1000 Epochs")
    plt.show()

def plot_loss_history():
    critic_loss = history_df['critic_loss'].to_numpy()
    actor_loss = history_df['actor_loss'].to_numpy()
    entropy = history_df['entropy'].to_numpy()

    x = np.arange(0,len(actor_loss)+1,step = 100)
    average_actor = []
    for i in range(len(x)-1):
        average_actor.append(np.nanmean(actor_loss[x[i]:x[i+1]]))

    average_critic = []
    for i in range(len(x)-1):
        average_critic.append(np.nanmean(critic_loss[x[i]:x[i+1]]))

    average_loss = []
    for i in range(len(x)-1):
        average_loss.append(np.nanmean(loss[x[i]:x[i+1]]))

    average_entropy = []
    for i in range(len(x)-1):
        average_entropy.append(np.nanmean(entropy[x[i]:x[i+1]]))

    fig,(ax1,ax2,ax3,ax4) = plt.subplots(4)
    ax1.plot(np.arange(len(entropy)),entropy)
    ax1.plot(x[1:],average_entropy)
    ax1.set_ylabel("Entropy")

    ax2.plot(np.arange(len(actor_loss)),actor_loss)
    ax2.plot(x[1:],average_actor)
    ax2.set_ylabel("Actor Loss")

    ax3.plot(np.arange(len(critic_loss)),critic_loss)
    ax3.plot(x[1:],average_critic)
    ax3.set_ylabel("Critic Loss")

    ax4.plot(np.arange(len(loss)),loss)
    ax4.plot(x[1:],average_loss)
    ax4.set_ylabel("Loss")
    plt.show()

def plot_trend_intersections_rsu():
    len_rsu_net = np.zeros(shape = ts_path_history.shape)
    for i in range(len(ts_path_history)):
        results = re.findall(r"\d+",ts_path_history[i])
        len_rsu_net[i] = len(results)

    len_intersections = np.zeros(shape = intersections_history.shape)
    for i in range(len(intersections_history)):
        results = np.array(re.findall(r"\d+",intersections_history[i])).astype(int)
        len_intersections[i] = len(results[results>0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(len_intersections,len_rsu_net,loss)
    ax.set_xlabel("Number of Intersections")
    ax.set_ylabel("Number of RSU")
    ax.set_zlabel("Loss")
    plt.show()

def plot_reward_critic_reward():
    lmbda = 0.24098677879102673
    reward = np.empty(shape = reward_history.shape,dtype=object)
    for i in range(reward.shape[0]):
        results = re.findall(r'[-+]?\d*\.?\d+',reward_history[i])
        reward[i] = [float(x) for x in results]


    critic_reward = np.empty(shape = reward_history.shape,dtype=object)
    for i in range(critic_reward.shape[0]):
        results = re.findall(r'[-+]?\d*\.?\d+',critic_reward_history[i])
        critic_reward[i] = [float(x) for x in results]

    reward = np.concatenate(reward).ravel()
    reward = reward[reward!=.1]
    critic_reward = np.concatenate(critic_reward).ravel()

    error = np.power(np.subtract(reward,critic_reward),2)

    # x = np.arange(len(error))
    # width = .35

    fig, (ax1,ax2,ax3) = plt.subplots(3)
    ax1.plot(reward)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Reward")
    ax2.plot(critic_reward)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Critic Reward")
    ax3.plot(error)
    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Squared Error")
    plt.show()

def plot_critic_performance():
    lmbda = 0.24098677879102673
    critic_loss = history_df['critic_loss'].to_numpy()
    average_critic = []
    x_loss = np.arange(0,len(critic_loss)+1,step = 100)
    for i in range(len(x_loss)-1):
        average_critic.append(np.nanmean(critic_loss[x_loss[i]:x_loss[i+1]]))

    critic_reward = np.empty(shape = reward_history.shape,dtype=object)
    for i in range(critic_reward.shape[0]):
        results = re.findall(r'[-+]?\d*\.?\d+',critic_reward_history[i])
        critic_reward[i] = [float(x) for x in results]

    critic_reward = np.concatenate(critic_reward).ravel()
    critic_reward = special.inv_boxcox(critic_reward, lmbda)
    x_reward = np.arange(len(critic_reward))

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.scatter(np.arange(len(critic_loss)),critic_loss)
    ax1.plot(x_loss[1:],average_critic)
    ax1.set_ylabel("Critic Loss")

    ax2.scatter(x_reward,critic_reward)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Critic Reward")
    plt.show()

def hist_rewards():
    lmbda = 0.24098677879102673
    reward = np.empty(shape = reward_history.shape,dtype=object)
    for i in range(reward.shape[0]):
        results = re.findall(r'[-+]?\d*\.?\d+',reward_history[i])
        reward[i] = [float(x) for x in results]


    critic_reward = np.empty(shape = reward_history.shape,dtype=object)
    for i in range(critic_reward.shape[0]):
        results = re.findall(r'[-+]?\d*\.?\d+',critic_reward_history[i])
        critic_reward[i] = [float(x) for x in results]

    reward = np.concatenate(reward).ravel()
    reward = reward[reward!=.1]
    critic_reward = np.concatenate(critic_reward).ravel()

    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.hist(reward)
    ax2.hist(critic_reward)
    plt.show()

def sqrt_rewards():
    reward = np.empty(shape = reward_history.shape,dtype=object)
    for i in range(reward.shape[0]):
        results = re.findall(r'[-+]?\d*\.?\d+',reward_history[i])
        reward[i] = [float(x) for x in results]
    reward = np.concatenate(reward).ravel()
    reward = reward[reward!=.1]
    sqrt_reward = np.sqrt(reward)

    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.hist(reward,bins=15)
    ax2.hist(sqrt_reward,bins=15)
    plt.show()


plot_single_loss()