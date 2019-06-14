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


class AgentActions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


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
        self.timestep_reward = []
        # self.rl_agent = DQNAgent(input_dim, output_dim)

    def init_q(self, s, a, type="ones"):
        """
        @param s the number of states
        @param a the number of actions
        @param type random, ones or zeros for the initialization
        """
        if type == "ones":
            return np.ones((s, a))
        elif type == "random":
            return np.random.random((s, a))
        elif type == "zeros":
            return np.zeros((s, a))


    def epsilon_greedy(self, epsilon, n_actions, s, train=False):
        """
        @param Q Q values state x action -> value
        @param epsilon for exploration
        @param s number of states
        @param train if true then no random actions selected
        """
        if train or np.random.rand() < epsilon:
            action = np.argmax(self.Q[s, :])
        else:
            action = np.random.randint(0, n_actions)
        return action

    def act(self, state):
        pass
    
    def process_reward(self, reward, s, s_, done):
        self.total_reward += reward
        a_ = np.argmax(Q[s_, :])
        if done:
            self.Q[s, a] += alpha * (reward - self.Q[s, a])
        else:
            self.Q[s, a] += alpha * (reward + (gamma * self.Q[s_, a_]) - self.Q[s, a])
        s, a = s_, a_
        if done:
            if render:
                print(f"This episode took {t} timesteps and reward: {self.total_reward}")
            timestep_reward.append(total_reward)
            break
