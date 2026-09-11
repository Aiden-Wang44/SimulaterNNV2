import numpy as np

from Math.Geometry.GeometryPrimitives import Trajectory
from SimulatedAlgorithms.localizer import *
from DataModels.robot import *
from Math.Geometry import *
class FeedForward:
    def __init__(self, ka: float, ks: float, kv: float):
        self.kA = ka
        self.kS = ks
        self.kV = kv
        self.prev_target_velocity = 0.0
    def get_target_voltage(self, target_velocity: float):
        rpm_velocity = target_velocity * 60.0
        sign_velocity = np.sign(rpm_velocity)
        if rpm_velocity == 0:
            sign_velocity = 0

        target_voltage = self.kS * sign_velocity + self.kV * rpm_velocity + self.kA * (rpm_velocity - self.prev_target_velocity)

        self.prev_target_velocity = rpm_velocity
        return target_voltage
class Ramsete:
    def __init__(self,localizer: SimpleLocalizer, robot: Robot, trajectory: Trajectory, beta: float, zeta: float   ):
        self.localizer = localizer
        self.robot = robot
        self.trajectory = trajectory
        self.beta = beta
        self.zeta = zeta
    def sinc(self, x: float):
        if(x == 0.0):
            return 1.0
        return math.sin(x)/x
    def update(self, time: float):
        self.localizer.update()
        target = self.trajectory.sample_by_time(time)
        p = self.localizer.get_pose()
        global_error = target.pose.point - p.point
        local_error = global_error.rotate(-p.heading.radians)
        heading_error = (target.pose.heading - p.heading).normalize()
        vd = target.linearVelocity
        wd = target.angularVelocity
        k = 2.0 * self.zeta * math.sqrt((wd**2) +self.beta * (vd**2) )
        v = vd * math.cos(heading_error.radians) + k * local_error.y
        w = wd + k * heading_error.radians - self.beta * vd * self.sinc(heading_error.radians) * local_error.x
        self.robot.drive_train.driveVelocity(v, 0 , w)
