import os, sys
import numpy as np
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)
from RSU_RL_System import DRL_System
from Actor_Critic import Actor_Critic, Actor, Critic
from Models.Agent import Agent
from Models.RSU_Intersections_Datamodule import RSU_Intersection_Datamodule

simulation_agent = Agent()
model = DRL_System(simulation_agent)
datamodule = RSU_Intersection_Datamodule(simulation_agent)

intersections,intersection_idx,rsu_net,mask = next(iter(datamodule.train_dataloader()))

def test_batch_sizes():
    assert(intersections.shape[0] == datamodule.batch_size)
    assert(intersections.shape[1] == datamodule.max_intersections + 1)
    assert(intersections[:,:,1:].shape[2] == model.num_features)

    assert(intersection_idx.shape[0] == datamodule.batch_size)
    assert(intersection_idx.shape[1] == datamodule.max_intersections+1)

    assert(rsu_net.shape[0] == datamodule.batch_size)
    assert(rsu_net.shape[1] == datamodule.max_pre_rsu_network)

    assert(mask.shape[0] == datamodule.batch_size)
    assert(mask.shape[1] == datamodule.max_intersections+1)

def test_actor_outputs():
    actor = model.actor_critic.actor
    log_probs, pointer_argmaxs = actor(intersections[:,:,1:],mask)

    assert(log_probs.shape[0] == datamodule.batch_size)
    assert(log_probs.shape[1] == datamodule.max_intersections+1)
    assert(log_probs.shape[2] == datamodule.max_intersections+1)
    assert(log_probs<=0).any() # The log(prob) are all within the valid range of values

    assert(pointer_argmaxs.shape[0] == datamodule.batch_size)
    assert(pointer_argmaxs.shape[1] == datamodule.max_intersections+1)
    assert(pointer_argmaxs>=0).any() # The indices are all possible values
    assert(pointer_argmaxs.max() <= datamodule.max_intersections+1) # The max value for an index is the last intersection considered

def test_critic_outputs():
    critic = model.actor_critic.critic
    critic_rewards = critic(intersections[:,:,1:])

    assert(critic_rewards.shape[0] == datamodule.batch_size)
    assert(critic_rewards.shape[1] == intersections.shape[1])

test_batch_sizes()
test_actor_outputs()
test_critic_outputs()

print(model)