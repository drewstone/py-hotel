from enum import Enum
import numpy as np


class AgentType(Enum):
    Simple = 1
    Random = 2
    Unknown = 3


class Error(Enum):
    UnexpectedNoneType = 1
    InvalidAgentType = 2
    Unknown = 3
    UnsupportedDimension = 4


class Agent(object):
    """docstring for Agent"""
    def __init__(self, agent_id, position, velocity):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.position = np.array(position)
        self.velocity = np.array(velocity)
    
    def move(self, state):
        self.position = state[self.agent_id]
        return self.position

    def process_reward(self, reward):
        print("Reward: {}, Position: {}".format(reward, self.position))
        pass

