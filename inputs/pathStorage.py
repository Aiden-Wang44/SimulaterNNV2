from Math.Geometry.GeometryPrimitives import *


class Parser:
    def __init__(self, path: str):
        self.path = path
        self.trajectory = Trajectory([])
        self.stripped_path = Path([])
    def parse(self, full_trajectory: bool = False):
        with open(self.path, 'r') as file:
            lines = file.readlines()

        for line in lines[1:]:
            data = line.split(',')
            time = float(data[0])
            x = float(data[1])
            y = float(data[2])
            if full_trajectory:
                heading = Angle(float(data[3]))
                linear_velocity = float(data[4])
                angular_velocity = float(data[5])
                trajectory_point = TrajectoryPoint(time,Pose(Vector2D(x,y),heading),linear_velocity,angular_velocity)
                self.trajectory.trajectory.append(trajectory_point)
            else:
                path_point = Vector2D(x,y)
                self.stripped_path.points.append(path_point)
        if full_trajectory:
            return self.trajectory
        else:
            return self.stripped_path

