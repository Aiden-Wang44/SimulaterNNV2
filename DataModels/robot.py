from Math.Geometry.GeometryPrimitives import Vector2D, Pose, Angle
import numpy as np
import math
from SimulatedAlgorithms.ramsete import FeedForward
from DataModels.frictionForces import FrictionForce
NM_TO_KG_IN2_S2 = 1550.3
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
        self.inertia =  0.32*mass * (width ** 2 + length ** 2) / 12
        self.drive_train = drive_train
        self.horizontal_tracker = horizontal_tracker
        self.vertical_tracker = vertical_tracker
        self.inertial = inertial


class DriveTrain:
    def __init__(self,left_gear_train: GearTrain,right_gear_train, drive_wheel: DriveWheel, left_motors: list[Motor], right_motors: list[Motor], track_width: float, feed_forward : FeedForward):
        self.left_gear_train = left_gear_train
        self.right_gear_train = right_gear_train
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
            motor.set_velocity(self.left_gear_train.get_velocity())
            left_torque+= motor.get_torque()

        return self.left_gear_train.get_torque(left_torque)
    def get_right_torque(self):
        right_torque = 0
        for motor in self.right_motors:
            motor.set_velocity(self.right_gear_train.get_velocity())
            right_torque+= motor.get_torque()

        return self.right_gear_train.get_torque(right_torque)
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
        speed = abs(self.velocity)

        if speed < 390:
            speed_factor = 1.0
        else:
            speed_factor = 1.0 - (speed - 390.0) / (600.0 - 390.0)

        speed_factor = max(0.0, speed_factor)
        return (
                0.33
                * NM_TO_KG_IN2_S2
                * (self.voltage / 12.0)
                * speed_factor
        )
    def set_velocity(self, velocity: float):
        self.velocity = velocity
#torque to torque
#gear ratio is torque/in over torque/out
class GearTrain:
    def __init__(self, gear_ratio: float, efficiency: float, inertia: float):
        self.gear_ratio = gear_ratio
        self.efficiency = efficiency
        # velocity in rad/s internally
        self.motor_velocity = 0.0
        self.inertia = inertia

    def get_torque(self, input_torque: float) -> float:
        """Pure query function: calculates output torque without mutating state."""
        return input_torque * self.efficiency / self.gear_ratio

    def update(self, input_torque: float, dt: float):
        """Explicit state update step: call this once per timestep."""

        accel = input_torque * self.efficiency / self.inertia
        self.motor_velocity += accel * dt
    def set_velocity(self,velocity):
        self.motor_velocity = velocity
    # returns rpm
    def get_velocity(self) -> float:
        return self.motor_velocity * 60.0 / (2 * math.pi)
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
    def __init__(self, radius: float):
        self.radius = radius
        self.rotation = 0.0
        self.velocity = 0.0
    def reset(self):
        self.rotation = 0.0
    def get_position(self):
        return self.rotation * self.radius

    def update(self, chassis_linear_vel: float, dt: float):
        target_velocity = chassis_linear_vel

        self.velocity = target_velocity
        self.rotation += (self.velocity / self.radius) * dt
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
    def __init__(self, robot: Robot, timestep: float, lateral_friction: FrictionForce, rotational_friction: FrictionForce, forward_friction: FrictionForce):
        self.robot = robot
        self.local_vel = Vector2D(0, 0)  # x = right, y = forward
        self.angular_vel = 0.0
        self.dt = timestep
        self.lateral_friction = lateral_friction
        self.rotational_friction = rotational_friction
        self.forward_friction = forward_friction
    def update(self):
        left_force = self.robot.drive_train.get_left_force()
        right_force = self.robot.drive_train.get_right_force()

        forward_force = left_force + right_force
        forward_friction_force = self.forward_friction.get_force(self.local_vel.y, forward_force, 2300) - self.local_vel.y*140
        required_lateral_force = self.robot.mass * self.angular_vel * self.local_vel.y

        lateral_force = self.lateral_friction.get_force(
            self.local_vel.x,
            required_lateral_force,
            self.robot.weight
        )


        # 2. Body-frame accelerations
        # a_x (Lateral / Right)

        lateral_accel = (lateral_force / self.robot.mass) + (self.angular_vel * self.local_vel.y)
        # a_y (Forward)

        forward_accel = (forward_force + forward_friction_force / self.robot.mass) - (self.angular_vel * self.local_vel.x)
        r = self.robot.drive_train.track_width / 2.0
        driving_torque = (right_force - left_force) * r
        required_rot_force = (right_force - left_force)
        rotational_surface_speed = self.angular_vel * r

        friction_force = self.rotational_friction.get_force(rotational_surface_speed,required_rot_force,self.robot.weight)
        friction_torque = friction_force * r

        # get_force() already returns opposite sign (-), so we simply add it
        net_torque = driving_torque + friction_torque
        angular_accel = net_torque / self.robot.inertia
        local_accel = Vector2D(lateral_accel, forward_accel)

        # 3. Integrate velocities & position
        self.local_vel = self.local_vel + local_accel * self.dt
        self.angular_vel = self.angular_vel + angular_accel * self.dt

        left_shaft_vel = (self.local_vel.y -self.angular_vel * self.robot.drive_train.track_width/2) / (self.robot.drive_train.drive_wheel.radius * self.robot.drive_train.left_gear_train.gear_ratio)
        right_shaft_vel = (self.local_vel.y +self.angular_vel * self.robot.drive_train.track_width/2) / (self.robot.drive_train.drive_wheel.radius * self.robot.drive_train.right_gear_train.gear_ratio)
        self.robot.drive_train.left_gear_train.set_velocity(left_shaft_vel)
        self.robot.drive_train.right_gear_train.set_velocity(right_shaft_vel)
        global_vel = self.local_vel.rotate(self.robot.pose.heading.radians)
        self.robot.pose.point = self.robot.pose.point + global_vel * self.dt
        self.robot.pose.heading = (self.robot.pose.heading + Angle(self.angular_vel) * self.dt).normalize()

        # Trackers (x = horizontal/lateral, y = vertical/forward)
        self.robot.horizontal_tracker.update(self.local_vel.x, self.dt)
        self.robot.vertical_tracker.update(self.local_vel.y, self.dt)
        self.robot.inertial.update(angular_accel, self.dt)