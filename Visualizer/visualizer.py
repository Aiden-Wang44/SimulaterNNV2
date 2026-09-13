import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


import math
import numpy as np
from DataModels.robot import *
from Math.Geometry.GeometryPrimitives import *
from matplotlib.patches import Polygon
@dataclass
class Settings:
    view_width: float = 80.0
    view_height: float = 80.0
    follow_robot: bool = True


class Visualizer:
    def __init__(self, robot: Robot ,path: Path, settings: Settings ,time_step:float):
        self.robot = robot
        self.settings = settings
        self.path = path
        self.time_step = time_step
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel="x axis", ylabel="y axis", title="robot visualization")
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.path_plot = self.ax.scatter(
            [point.x for point in self.path.points],
            [point.y for point in self.path.points]
        )
        self.robot_patch = Polygon(np.empty((0, 2)), closed=True)
        self.ax.add_patch(self.robot_patch)
        self.heading_line, = self.ax.plot([], [], linewidth=2)

        self.update_view()

    # Local coordinate system:
    # +Y = forward
    # +X = right
    def update_robot(self):
        x = self.robot.pose.point.x
        y = self.robot.pose.point.y
        heading = self.robot.pose.heading
        half_width = self.robot.width/2
        half_length = self.robot.length/2
        corners = [
            (-half_width, half_length),
            (half_width, half_length),
            (half_width, -half_length),
            (-half_width, -half_length),
        ]
        cos_h = math.cos(heading.radians)
        sin_h = math.sin(heading.radians)
        world_corners = []
        for local_x, local_y in corners:
            world_x = x  + local_x * cos_h - local_y * sin_h
            world_y = y + local_x * sin_h + local_y * cos_h
            world_corners.append([world_x, world_y])
        self.robot_patch.set_xy(world_corners)
        arrow_length = self.robot.length / 1.5
        end_x = x - arrow_length * sin_h
        end_y = y + arrow_length * cos_h
        self.heading_line.set_data([x,end_x], [y,end_y])

    def update_view(self):
        if not self.settings.follow_robot:
            return
        x = self.robot.pose.point.x
        y = self.robot.pose.point.y
        self.ax.set_xlim(x - self.settings.view_width/2, x + self.settings.view_width/2)
        self.ax.set_ylim(y - self.settings.view_height/2, y+self.settings.view_height/2)

    def show(self):
        plt.show()

    def draw(self):
        self.update_robot()
        self.update_view()
        self.fig.canvas.draw_idle()
        plt.pause(self.time_step)



