from context import agent, sim
import unittest


class TestAgentMethods(unittest.TestCase):

    def test_sim_constructor(self):
        num_agents = 1
        agent_type = agent.AgentType.Simple
        state_space_dim = 1
        s = sim.Simulator(num_agents, agent_type, state_space_dim)
        self.assertEqual(len(s.agents), num_agents)
        self.assertEqual(len(s.agents[0].position), state_space_dim)
        self.assertEqual(len(s.agents[0].velocity), state_space_dim)

    def test_sim_step(self):
        num_agents = 10
        agent_type = agent.AgentType.Random
        state_space_dim = 2
        s = sim.Simulator(num_agents, agent_type, state_space_dim)
        s.step()

if __name__ == '__main__':
    unittest.main()
