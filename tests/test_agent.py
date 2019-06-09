from context import agent
import unittest


class TestAgentMethods(unittest.TestCase):

    def test_agent_constructor(self):
        a = agent.Agent(0, [0.0], [0.0])
        self.assertEqual(a.agent_id, 0)

if __name__ == '__main__':
    unittest.main()
