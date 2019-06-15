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


class AgentActions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Agent(object):
    """
    Basic class for an Agent
    """

    def __init__(
        self,
        agent_id,
        position,
        input_dim,
        output_dim,
        bounding_box
    ):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.position = np.array(position)
        self.timestep_reward = []
        self.min_x = bounding_box[0]
        self.max_x = bounding_box[1]
        self.min_y = bounding_box[2]
        self.max_y = bounding_box[3]

    def act(self, state):
        action = np.random.choice(list(AgentActions))
        new_position = self.process_action(action)
        out_of_bounds_vert = new_position[1] > self.max_y or new_position[1] < self.min_y
        out_of_bounds_hori = new_position[0] > self.max_x or new_position[0] < self.min_x
        # while still out of bounds in any direction, select new action
        while out_of_bounds_vert or out_of_bounds_hori:
            action = np.random.choice(list(AgentActions))
            new_position = self.process_action(action)
            out_of_bounds_vert = new_position[1] > self.max_y or new_position[1] < self.min_y
            out_of_bounds_hori = new_position[0] > self.max_x or new_position[0] < self.min_x

        self.position = new_position
        return self.position

    def process_action(self, action):
        if action == AgentActions.UP:
            return np.array([self.position[0], self.position[1] + 1])
        elif action == AgentActions.DOWN:
            return np.array([self.position[0], self.position[1] - 1])
        elif action == AgentActions.LEFT:
            return np.array([self.position[0] - 1, self.position[1]])
        elif action == AgentActions.RIGHT:
            return np.array([self.position[0] + 1, self.position[1]])

    def process_reward(self, reward, s, s_, done):
        pass


class QLearningAgent(Agent):
    """
    Basic class for an Agent
    """

    def __init__(
        self,
        agent_id,
        position,
        input_dim,
        output_dim,
        num_states,
        num_actions,
    ):
        super().__init__(agent_id, position, input_dim, output_dim)
        self.Q = self.init_q(num_states, num_actions)

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


def create_agents(num_agents, agent_type, ss_dim, bounding_box):
    agents = []
    min_x = bounding_box[0]
    max_x = bounding_box[1]
    min_y = bounding_box[2]
    max_y = bounding_box[3]
    for i in range(num_agents):
        if agent_type == AgentType.Simple:
            # Simple agents all start in center of [0,1]
            agent = Agent(
                i,
                np.full((ss_dim,), int(min(np.avg(min_x, max_x), np.avg(min_y, max_y)))),
                ss_dim * num_agents,
                ss_dim,
                bounding_box)
        elif agent_type == AgentType.Random:
            agent = Agent(
                i,
                np.random.randint(min(max_x, max_y), size=ss_dim),
                ss_dim * num_agents,
                ss_dim,
                bounding_box)
        else:
            raise ValueError(Error.InvalidAgentType)

        agents.append(agent)
    return agents
