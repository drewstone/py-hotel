import sys
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
            volumes, regions, vertices = self.process_vor(vor)
            total_volume = sum(list(map(lambda elt: elt["volume"], volumes)))
            # log deviation from full volume
            if total_volume != 10000.0:
                print("Total vol: {}".format(total_volume))
            # re-match up volumes to points by their centroids
            for inx, elt in enumerate(self.agents):
                min_inx = 0
                min_dist = 9223372036854775807.0
                for iinx, eelt in enumerate(volumes):
                    dist = np.linalg.norm(elt.position - eelt["centroid"])
                    if dist < min_dist:
                        min_dist = dist
                        min_inx = iinx

                self.agents[inx].process_reward(volumes[min_inx]["volume"])

            try:
                self.plot_voronoi(vor, regions, vertices, unique_actions)
            except Exception as e:
                raise e
        else:
            raise ValueError(Error.UnsupportedDimension)

    def process_vor(self, vor):
        regions, vertices = voronoi_finite_polygons_2d(vor)

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
            poly = poly.intersection(box)
            polygon = [p for p in poly.exterior.coords]
            new_vertices = np.stack(polygon)
            hull = ConvexHull(new_vertices)
            # centoid
            cx = np.mean(hull.points[hull.vertices, 0])
            cy = np.mean(hull.points[hull.vertices, 1])
            centroid = np.array([cx, cy])
            # append volume and centroid
            volumes.append({"volume": hull.volume, "centroid": centroid});
        return volumes, regions, vertices

    def create_agent(self, agent_id, agent_type, state_space_dim):
        mul_factor = 1e2
        if agent_type == AgentType.Simple:
            # Simple agents all start in center of [0,1] with largest velocity
            position = np.around(np.full((state_space_dim,), 0.5) * mul_factor, decimals=3)
            velocity = np.around(np.full((state_space_dim,), 1.0) * mul_factor, decimals=3)
            return Agent(agent_id, position, velocity)
        elif agent_type == AgentType.Random:
            position = np.around(np.random.uniform(0, 1, state_space_dim) * mul_factor, decimals=3)
            velocity = np.around(np.random.uniform(0, 1, state_space_dim) * mul_factor, decimals=3)
            return Agent(agent_id, position, velocity)
        else:
            raise ValueError(Error.InvalidAgentType)

    def plot_voronoi(self, vor, regions, vertices, points):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

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
            poly = Polygon(polygon)
            poly = poly.intersection(box)
            polygon = [p for p in poly.exterior.coords]
            ax.fill(*zip(*polygon), alpha=0.4)

        ax.plot(points[:, 0], points[:, 1], 'ko')
        ax.axis('equal')
        # ax.xlim(-200, 200)
        # ax.ylim(-200, 200)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 101, 20)
        minor_ticks = np.arange(0, 101, 5)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='major', color='b', linestyle='--')
        ax.grid(which='minor', color='r', linestyle=':')
        plt.savefig('voro.png', dpi=200)

if __name__ == '__main__':
    num_agents = 10
    agent_type = AgentType.Random
    state_space_dim = 2
    bounding_box = np.array([0.0, 1.0, 0.0, 0.0])  # [x_min,x_max,y_min,y_max]
    s = Simulator(num_agents, agent_type, state_space_dim, bounding_box)
    s.step()
