import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from agent import Agent, AgentType, Error
from voronoi_util import voronoi_finite_polygons_2d
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon

MAX_VALUE = 9223372036854775807.0


class Simulator(object):
    """docstring for Simulator"""
    def __init__(
        self,
        num_agents,
        agent_type,
        ss_dim,
        bounding_box,
        episode_length=100
    ):
        super(Simulator, self).__init__()
        self.ss_dim = ss_dim
        self.round = 0
        self.agents = self.create_agents(num_agents, agent_type, ss_dim)
        self.min_x = bounding_box[0]
        self.max_x = bounding_box[1]
        self.min_y = bounding_box[2]
        self.max_y = bounding_box[3]
        self.episode_length = episode_length
        self.colors = [np.random.rand(3,) for i in range(num_agents)]

    def run_episode(self):
        for i in range(self.episode_length):
            # stack state of environment
            positions = np.stack(list(map(
                lambda a: a.position,
                self.agents)))
            # stack actions of agents
            actions = np.stack(list(map(
                lambda a: a.act(positions),
                self.agents)))
            rewards = self.step(actions)

            for inx, agent in enumerate(self.agents):
                agent.remember(
                    positions,      # state
                    actions[inx],   # action
                    rewards[inx],   # reward
                    actions,        # next_state
                    False)          # done

        for agent in self.agents:
            agent.replay(32)

    def step(self, actions):
        # order agents
        rewards, region_indices = [], []
        # compute unique actions for tesselation
        unique_actions = np.unique(actions, axis=0)
        # compute Voronoi tesselation
        vor = Voronoi(unique_actions)
        if self.ss_dim == 2:
            volumes, regions, vertices = self.process_vor(vor)
            # re-match up volumes to points by their centroids
            for _, agent in enumerate(self.agents):
                min_inx = 0
                min_dist = MAX_VALUE
                # match each point to closest centroid by l2-norm
                for inx, elt in enumerate(volumes):
                    dist = np.linalg.norm(agent.position - elt["centroid"])
                    if dist < min_dist:
                        min_dist = dist
                        min_inx = inx
                # append correct reward to agent index
                rewards.append(volumes[min_inx]["volume"])
                # append correct region index to agent index
                region_indices.append(min_inx)
            self.plot_voronoi(vor, regions, vertices, unique_actions, region_indices)
            return rewards
        else:
            raise ValueError(Error.UnsupportedDimension)

    def process_vor(self, vor):
        regions, vertices = voronoi_finite_polygons_2d(vor)
        box = self.get_bounding_box(vertices)
        volumes = []
        for inx, region in enumerate(regions):
            polygon = vertices[region]
            # Clipping polygon
            poly = Polygon(polygon)
            poly = poly.intersection(box)
            polygon = [p for p in poly.exterior.coords]
            hull = ConvexHull(np.stack(polygon))
            centroid = np.array([
                np.mean(hull.points[hull.vertices, 0]),
                np.mean(hull.points[hull.vertices, 1])
            ])
            # append volume and centroid
            volumes.append({"volume": hull.volume, "centroid": centroid})
        return volumes, regions, vertices

    def create_agents(self, num_agents, agent_type, ss_dim):
        agents = []
        for i in range(num_agents):
            if agent_type == AgentType.Simple:
                # Simple agents all start in center of [0,1]
                agent = Agent(
                    i,
                    np.around(np.full((ss_dim,), 50.0), decimals=3),
                    ss_dim * num_agents,
                    ss_dim)
            elif agent_type == AgentType.Random:
                agent = Agent(
                    i,
                    np.around(np.random.uniform(0, 100.0, ss_dim), decimals=3),
                    ss_dim * num_agents,
                    ss_dim)
            else:
                raise ValueError(Error.InvalidAgentType)

            agents.append(agent)
        return agents

    def get_agent_of_region_inx(self, region_indices, index):
        for inx, region_inx in enumerate(region_indices):
            if region_inx == index:
                return inx

    def get_agent_index_for_point(self, point):
        for inx, agent in enumerate(self.agents):
            if agent.position[0] == point[0] and agent.position[1] == point[1]:
                return inx

    def get_bounding_box(self, vertices):
        mins = np.tile((self.min_x, self.min_y), (vertices.shape[0], 1))
        bounded_vertices = np.max((vertices, mins), axis=0)
        maxs = np.tile((self.max_x, self.max_y), (vertices.shape[0], 1))
        bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

        return Polygon([
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y]])

    def plot_voronoi(self, vor, regions, vertices, points, region_indices):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        box = self.get_bounding_box(vertices)

        # colorize
        for inx, region in enumerate(regions):
            polygon = vertices[region]
            # Clipping polygon
            poly = Polygon(polygon)
            poly = poly.intersection(box)
            polygon = [p for p in poly.exterior.coords]
            # agent_inx = self.get_agent_of_region_inx(region_indices, inx)
            ax.fill(*zip(*polygon), alpha=0.4)

        ax.plot(points[:, 0], points[:, 1], 'ko')
        for inx, elt in enumerate(points):
            ax.annotate(
                self.get_agent_index_for_point(points[inx]),
                (points[inx][0], points[inx][1]))

        ax.axis('equal')
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 101, 20)
        minor_ticks = np.arange(0, 101, 5)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        # ax.grid(which='major', color='b', linestyle='--')
        # ax.grid(which='minor', color='r', linestyle=':')
        plt.savefig('voro.png', dpi=200)

if __name__ == '__main__':
    min_x = 0
    max_x = 100
    min_y = 0
    max_y = 100

    num_agents = 10
    agent_type = AgentType.Random
    ss_dim = 2
    bounding_box = np.array([min_x, max_x, min_y, max_y])
    s = Simulator(
        num_agents,
        agent_type,
        ss_dim,
        bounding_box,
        episode_length=64)
    s.run_episode()
