import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as animation

from agent import Agent, AgentType, Error, create_agents
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
        episode_length=100,
        ax=None
    ):
        super(Simulator, self).__init__()
        self.ss_dim = ss_dim
        self.round = 0
        self.agents = create_agents(
            num_agents,
            agent_type,
            ss_dim,
            bounding_box)
        self.min_x = bounding_box[0]
        self.max_x = bounding_box[1]
        self.min_y = bounding_box[2]
        self.max_y = bounding_box[3]
        self.episode_length = episode_length
        self.colors = [np.random.rand(3,) for i in range(num_agents)]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def step(self, i=0):
        # stack state of environment
        positions = np.stack(list(map(
            lambda a: a.position,
            self.agents)))
        # stack actions of agents
        actions = np.stack(list(map(
            lambda a: a.act(positions),
            self.agents)))
        # return results of processing step
        region_data, regions, vertices = self.process_step(actions)
        # plot data
        im = self.vor_fills(region_data, regions, vertices, actions)
        self.ax.axis('equal')
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 101, 20)
        minor_ticks = np.arange(0, 101, 5)

        self.ax.set_xticks(major_ticks)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(major_ticks)
        self.ax.set_yticks(minor_ticks, minor=True)
        # return all fills and points
        return im

    def process_step(self, actions):
        # order agents
        rewards, region_indices = [], []
        # compute Voronoi tesselation
        vor = Voronoi(actions)
        # split by state space dimension
        if self.ss_dim == 2:
            return self.process_vor(vor)
        else:
            raise ValueError(Error.UnsupportedDimension)

    def process_vor(self, vor):
        regions, vertices = voronoi_finite_polygons_2d(vor)
        box = self.get_bounding_box()
        region_data = []
        for inx, region in enumerate(regions):
            # Clipping polygon
            poly = Polygon(vertices[region]).intersection(box)
            polygon = [p for p in poly.exterior.coords]
            points = np.stack(polygon)
            hull = ConvexHull(points)
            # match agent with region data
            for a_inx, a in enumerate(self.agents):
                h_path = Path(points[hull.vertices])
                if h_path.contains_point(a.position):
                    region_data.append({
                        "polygon": polygon,
                        "reward": hull.volume,
                        "agent_inx": a_inx,
                        "agent_pos": a.position,
                        "region_inx": inx,
                    })
        return region_data, regions, vertices

    def get_bounding_box(self):
        return Polygon([[self.min_x, self.min_y], [self.min_x, self.max_y],
                        [self.max_x, self.max_y], [self.max_x, self.min_y]])

    def vor_fills(self, region_data, regions, vertices, points):
        im = []
        box = self.get_bounding_box()
        self.ax.clear()
        for inx, region in enumerate(regions):
            poly = Polygon(vertices[region]).intersection(box)
            polygon = [p for p in poly.exterior.coords]
            fill = self.ax.fill(
                *zip(*polygon),
                alpha=0.4,
                c=self.colors[region_data[inx]["agent_inx"]])
            im.append(fill)
        pts = self.ax.plot(points[:, 0], points[:, 1], 'ko')
        im.append(pts)
        plt.savefig('voro.png', dpi=200)
        return im

    def start(self):
        self.anim = animation.FuncAnimation(
            self.fig,
            self.step,
            np.arange(1, 200),
            interval=1)
        self.anim.save('vor.mp4', writer='imagemagick', fps=30)

if __name__ == '__main__':
    min_x = 0
    max_x = 100
    min_y = 0
    max_y = 100

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    num_agents = 5
    agent_type = AgentType.Random
    ss_dim = 2
    bounding_box = np.array([min_x, max_x, min_y, max_y])
    s = Simulator(
        num_agents,
        agent_type,
        ss_dim,
        bounding_box,
        episode_length=5)
    s.start()
