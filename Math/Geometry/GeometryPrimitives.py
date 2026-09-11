from dataclasses import dataclass
import math
@dataclass
class Vector2D:
    x: float
    y: float
    def __sub__(self,other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)
    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)
    def __mul__(self, other: float) -> "Vector2D":
        return Vector2D(self.x * other, self.y * other)
    def __rmul__(self, other: float) -> "Vector2D":
        return self * other

    def rotate(self, theta: float) -> "Vector2D":
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        return Vector2D(
            self.x * cos_theta - self.y * sin_theta,
            self.x * sin_theta + self.y * cos_theta
        )
@dataclass(frozen = True)
class Angle:
    radians: float
    def normalize(self) -> "Angle":
        angle = (self.radians + math.pi) % (2 * math.pi) - math.pi
        return Angle(angle)
    def __sub__(self, other: "Angle") -> "Angle":
        return Angle(self.radians - other.radians)
    def __add__(self, other: "Angle") -> "Angle":
        return Angle(self.radians + other.radians)
    def __mul__(self, other: float) -> "Angle":
        return Angle(self.radians * other)
    def __rmul__(self, other: float) -> "Angle":
        return self * other
    def __neg__(self) -> "Angle":
        return Angle(-self.radians)
@dataclass
class Pose:
    point: Vector2D
    heading: Angle
@dataclass
class Path:
    points : list[Vector2D]
@dataclass
class TrajectoryPoint:
    time: float
    pose: Pose
    linearVelocity:float
    angularVelocity:float

class Trajectory:
    def __init__(self, trajectory : list[TrajectoryPoint]):
        self.trajectory = trajectory
    def sample_by_index(self, index):
        return self.trajectory[index]
    def sample_by_time(self,time):
        if time<= self.trajectory[0].time :
            return self.trajectory[0]
        elif time>= self.trajectory[-1].time:
            return self.trajectory[-1]
        for i in range(0, len(self.trajectory)-1,1):
            if self.trajectory[i].time <= time <= self.trajectory[i + 1].time:
                t0 = self.trajectory[i].time
                t1 = self.trajectory[i + 1].time
                ratio = (time-t0)/(t1-t0)
                p0 = self.trajectory[i]
                p1 = self.trajectory[i + 1]

                v = (p1.pose.point - p0.pose.point)*ratio + p0.pose.point
                heading = ((p1.pose.heading - p0.pose.heading).normalize()* ratio+ p0.pose.heading).normalize()
                linear_velocity = (p1.linearVelocity - p0.linearVelocity) * ratio + p0.linearVelocity
                angular_velocity = (p1.angularVelocity - p0.angularVelocity) * ratio + p0.angularVelocity
                return TrajectoryPoint(time, Pose(v,heading), linear_velocity, angular_velocity)
        return self.trajectory[-1]
    def get_size(self):
        return len(self.trajectory)
    def get_duration(self):
        return self.trajectory[-1].time -self.trajectory[0].time

