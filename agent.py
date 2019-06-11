from enum import Enum
import numpy as np
from models import DQNAgent


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
    def __init__(
        self,
        agent_id,
        position,
        input_dim,
        output_dim,
    ):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.position = np.array(position)
        self.rl_agent = DQNAgent(input_dim, output_dim)

    def act(self, state):
        # flatten state when passing to rl_agent
        new_position = self.rl_agent.act(state.flatten())
        self.position = new_position
        return self.position

    def replay(self, batch_size):
        return self.rl_agent.replay(batch_size)

    def remember(self, state, action, reward, next_state, done):
        return self.rl_agent.remember(
            state.flatten(),
            action,
            reward,
            next_state.flatten(),
            done)
