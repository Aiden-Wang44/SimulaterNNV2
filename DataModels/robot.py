from Math.Geometry.GeometryPrimitives import Vector2D, Pose, Angle
import numpy as np
import math
from SimulatedAlgorithms.ramsete import FeedForward
NM_TO_KG_IN2_S2 = 1550.003
class Robot:
    def __init__(
        self,
        width: float,
        length: float,
        pose: Pose,
        mass: float,
        drive_train : DriveTrain,
        horizontal_tracker: TrackerWheel,
        vertical_tracker: TrackerWheel,
        inertial: IMU,
    ):
        self.width = width
        self.length = length
        self.pose = pose
        self.mass = mass
        self.weight = mass * 386.09
        self.inertia = mass * (width ** 2 + length ** 2) / 12
        self.drive_train = drive_train
        self.horizontal_tracker = horizontal_tracker
        self.vertical_tracker = vertical_tracker
        self.inertial = inertial


class DriveTrain:
    def __init__(self,gear_train: GearTrain, drive_wheel: DriveWheel, left_motors: list[Motor], right_motors: list[Motor], track_width: float, feed_forward : FeedForward):
        self.gear_train = gear_train
        self.drive_wheel = drive_wheel
        self.left_motors = left_motors
        self.right_motors = right_motors
        self.track_width = track_width
        self.feed_forward = feed_forward
    def driveVelocity(self,forward_velocity: float, lateral_velocity, angular_velocity):
        left_velocity = (forward_velocity - angular_velocity * self.track_width/2)/ (self.drive_wheel.radius * 2 * math.pi)
        right_velocity = (forward_velocity + angular_velocity * self.track_width / 2) / (self.drive_wheel.radius * 2 * math.pi)
        self.set_left_voltage(self.feed_forward.get_target_voltage(left_velocity))
        self.set_right_voltage(self.feed_forward.get_target_voltage(right_velocity))
    def set_left_voltage(self, voltage: float):
        for motor in self.left_motors:
            motor.set_voltage(voltage)
    def set_right_voltage(self, voltage: float):
        for motor in self.right_motors:
            motor.set_voltage(voltage)

    def get_left_torque(self):
        left_torque = 0
        for motor in self.left_motors:
            left_torque+= motor.get_torque()
        return self.gear_train.get_torque(left_torque)
    def get_right_torque(self):
        right_torque = 0
        for motor in self.right_motors:
            right_torque+= motor.get_torque()
        return self.gear_train.get_torque(right_torque)
    def get_left_force(self):
        left_torque = self.get_left_torque()
        return self.drive_wheel.get_force(left_torque)
    def get_right_force(self):
        right_torque = self.get_right_torque()
        return self.drive_wheel.get_force(right_torque)

#voltage to torque
class Motor:
    def __init__(self):
        self.velocity = 0.0
        self.voltage = 0.0
    def set_voltage(self, voltage: float):
        if abs(voltage) > 12.0:
            self.voltage = np.sign(voltage) * 12.0
        else:
            self.voltage = voltage

    def get_torque(self):
        if self.voltage == 0:
            return 0.0

        return 0.35 * NM_TO_KG_IN2_S2 * (self.voltage / 12.0)
#torque to torque
#gear ratio is speed/in over speed/out
class GearTrain:
    def __init__(self,gear_ratio: float, efficiency: float):
        self.gear_ratio = gear_ratio
        self.efficiency = efficiency
    def get_torque(self, input_torque: float):
        return input_torque * self.efficiency / self.gear_ratio
#torque to force
class DriveWheel:
    def __init__(self, max_ground_force:float, radius:float):
        self.max_ground_force = max_ground_force
        self.radius = radius
    def get_force(self, input_torque: float):
        s = np.sign(input_torque)
        return s * min(abs(input_torque / self.radius), self.max_ground_force)
#dead trackerwheels
class TrackerWheel:
    def __init__(self, radius:float, max_ground_force: float, mass: float):
        self.max_ground_force = max_ground_force
        self.mass= mass
        self.radius = radius
        self.rotation = 0.0
        self.velocity = 0.0
        self.inertia = self.mass * (self.radius**2)/2
    def reset(self):
        self.rotation = 0.0
    def get_position(self):
        return self.rotation * self.radius
    def update(self, acceleration: float, dt: float):
        torque = np.sign(acceleration) * min(abs(acceleration *self.mass), self.max_ground_force ) * self.radius
        ang_accel = torque / self.inertia
        self.velocity += ang_accel * dt
        self.rotation += self.velocity * dt

class IMU:
    def __init__(self):
        self.heading = Angle(0.0)
        self.ang_vel = 0.0
    def set_heading(self, heading: Angle):
        self.heading = heading.normalize()
    def update(self, ang_accel: float, dt: float):
        self.ang_vel += ang_accel * dt
        self.heading = (self.heading + Angle(self.ang_vel * dt)).normalize()

    def get_heading(self):
        return self.heading

class ChassisDynamics:
    def __init__(self,robot: Robot, timestep : float):
        self.robot = robot
        self.local_vel = Vector2D(0,0)
        self.angular_vel = 0.0
        self.dt = timestep

    def update(self):
        left_force = self.robot.drive_train.get_left_force()
        right_force = self.robot.drive_train.get_right_force()

        forward_accel = (left_force + right_force) / self.robot.mass

        local_accel = Vector2D(-self.angular_vel * self.local_vel.y,forward_accel + self.angular_vel * self.local_vel.x)
        angular_accel = ((right_force - left_force) * self.robot.drive_train.track_width/2)/self.robot.inertia

        self.local_vel = self.local_vel + local_accel * self.dt
        self.angular_vel = self.angular_vel + angular_accel * self.dt
        global_vel = self.local_vel.rotate(self.robot.pose.heading.radians)
        self.robot.pose.point = self.robot.pose.point + global_vel * self.dt
        self.robot.pose.heading = (self.robot.pose.heading + Angle(self.angular_vel) * self.dt).normalize()
        self.robot.horizontal_tracker.update(local_accel.x, self.dt)
        self.robot.vertical_tracker.update(local_accel.y, self.dt)
        self.robot.inertial.update(angular_accel,self.dt)

