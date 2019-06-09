import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from agent import Agent, AgentType, Error
from voronoi_util import voronoi_finite_polygons_2d
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon


class Simulator(object):
    """docstring for Simulator"""
    def __init__(self, num_agents, agent_type, state_space_dim, bounding_box):
        super(Simulator, self).__init__()
        self.state_space_dim = state_space_dim
        self.round = 0
        self.agents = list(map(lambda i:
                           self.create_agent(i, agent_type, state_space_dim),
                           range(num_agents)))
        self.bounding_box = bounding_box

    def step(self):
        positions = np.stack(list(map(lambda a: a.position, self.agents)))
        actions = np.stack(list(map(lambda a: a.move(positions), self.agents)))
        unique_actions = np.unique(actions, axis=0)
        # compute Voronoi tesselation
        vor = Voronoi(unique_actions)

        if self.state_space_dim == 2:
            regions, vertices = voronoi_finite_polygons_2d(vor)

            try:
                self.plot_voronoi(vor, regions, vertices, unique_actions)
            except Exception as e:
                raise e

            min_x = 0
            max_x = 100
            min_y = 0
            max_y = 100

            mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
            bounded_vertices = np.max((vertices, mins), axis=0)
            maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
            bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

            box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
            volumes = []
            # colorize
            for inx, region in enumerate(regions):
                polygon = vertices[region]
                # Clipping polygon
                poly = Polygon(polygon)
                # poly = poly.intersection(box)
                polygon = [p for p in poly.exterior.coords]
                new_vertices = np.stack(polygon)
                volume = ConvexHull(new_vertices).volume
                print(new_vertices, self.agents[inx].position)
                self.agents[inx].process_reward(volume)
                volumes.append(volume)

            print(sum(volumes))
        else:
            raise ValueError(Error.UnsupportedDimension)

    def create_agent(self, agent_id, agent_type, state_space_dim):
        mul_factor = 1e2
        if agent_type == AgentType.Simple:
            # Simple agents all start in center of [0,1] with largest velocity
            position = np.full((state_space_dim,), 0.5) * mul_factor
            velocity = np.full((state_space_dim,), 1.0) * mul_factor
            return Agent(agent_id, position, velocity)
        elif agent_type == AgentType.Random:
            position = np.random.uniform(0, 1, state_space_dim) * mul_factor
            velocity = np.random.uniform(0, 1, state_space_dim) * mul_factor
            return Agent(agent_id, position, velocity)
        else:
            raise ValueError(Error.InvalidAgentType)

    def plot_voronoi(self, vor, regions, vertices, points):
        min_x = 0
        max_x = 100
        min_y = 0
        max_y = 100

        mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
        bounded_vertices = np.max((vertices, mins), axis=0)
        maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
        bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

        box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

        # colorize
        for region in regions:
            polygon = vertices[region]
            # Clipping polygon
            # poly = Polygon(polygon)
            # poly = poly.intersection(box)
            # polygon = [p for p in poly.exterior.coords]
            plt.fill(*zip(*polygon), alpha=0.4)

        plt.plot(points[:, 0], points[:, 1], 'ko')
        plt.axis('equal')
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)

        plt.savefig('voro.png')

if __name__ == '__main__':
    num_agents = 4
    agent_type = AgentType.Random
    state_space_dim = 2
    bounding_box = np.array([0.0, 1.0, 0.0, 0.0])  # [x_min,x_max,y_min,y_max]
    s = Simulator(num_agents, agent_type, state_space_dim, bounding_box)
    s.step()
