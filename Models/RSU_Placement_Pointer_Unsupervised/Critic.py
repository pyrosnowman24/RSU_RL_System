import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from pytorch_lightning import LightningModule, Trainer

import numpy as np

class Embedding(nn.Module):
    def __init__(self,c_inputs,c_embed):
        super(Embedding,self).__init__()
        self.embedding_layer = nn.Linear(c_inputs, c_embed, bias=False)

    def forward(self,data):
        return self.embedding_layer(data)

class Critic(nn.Module):
    def __init__(self,num_features = 3,
                      embedding_size = 16 ):
        super(Critic,self).__init__()
        self.embedding = Embedding(num_features,embedding_size)
        self.network = nn.Sequential(nn.Linear(embedding_size,16),
                                     nn.ReLU(),
                                     nn.Linear(16,32),
                                     nn.ReLU(),
                                     nn.Linear(32,16),
                                     nn.ReLU(),
                                     nn.Linear(16,1))
        

    def forward(self,items):
        embedded_state = self.embedding(items)

        rewards = []
        for sample in embedded_state:
            reward = self.network(sample)
            rewards.append(reward)
        rewards = torch.stack(rewards,dim=0)

        return rewards[:,:,0]
    
class Critic_Lightning(LightningModule):
    def __init__(self, 
                 num_features = 2,
                 embedding_size = 16,
                ):
        super(Critic_Lightning,self).__init__()
        self.critic = Critic(num_features=num_features, embedding_size=embedding_size)

    def forward(self):
        return 1
    
    def training_step(self, *args, **kwargs):
        return 1
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initializes Adam optimizers for actor and critic.

        Returns:
            List[Optimizer]: Optimizers for actor and critic.
        """
        optimizer = Adam(self.critic.parameters(), lr=self.lr)
        return optimizer