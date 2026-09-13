import math
import time
from Math.Geometry.GeometryPrimitives import *
from inputs.pathStorage import *
from SimulatedAlgorithms.localizer import *
from SimulatedAlgorithms.ramsete import *
from DataModels.robot import *
from DataModels.robot import GearTrain
from Visualizer.visualizer import *
from DataModels.frictionForces import FrictionForce
time_step = 0.01
gear_train_l = GearTrain(0.75, 1, 0.374)
gear_train_r = GearTrain(0.75, 1, 0.374)
drive_wheel = DriveWheel(1000000,1.625)
left_motors = []
right_motors = []
for i in range (3):
    left_motors.append(Motor())
for i in range (3):
    right_motors.append(Motor())
horizontal_tracker = TrackerWheel(1.375)
vertical_tracker = TrackerWheel(1)
inertial = IMU()
feed_forward = FeedForward(0.52, 1.155, 0.02)
lateral_friction_force = FrictionForce(0.09,0.075)
forward_friction_force = FrictionForce(1,0.95)
rotational_friction_force = FrictionForce(0.95,0.94)
drive_train = DriveTrain(gear_train_l,gear_train_r,drive_wheel,left_motors,right_motors,11.5, feed_forward)
robot = Robot(11.5,12.0,Pose(Vector2D(0.0,0.0),Angle(0.0)),6.75,drive_train,horizontal_tracker,vertical_tracker,inertial)
odom = TwoAxisTwoWheelOdom(horizontal_tracker,vertical_tracker,inertial)
localizer = SimpleLocalizer(odom)
dynamics = ChassisDynamics(robot, time_step, lateral_friction_force, rotational_friction_force, forward_friction_force)
settings = Settings()
textParser = Parser("inputs/pathOne.txt")
path_one = textParser.parse()
path_one_full = textParser.parse(True)
visualizer = Visualizer(robot,path_one, settings ,time_step)
path_follower = Ramsete(localizer,robot,path_one_full,0.028,0.23)
print(path_one)
visualizer.draw()

# Test movement.
curr_time = 0

while curr_time < path_one_full.get_duration() :
    path_follower.update(curr_time)
    dynamics.update()
    visualizer.draw()
    curr_time+= time_step
visualizer.show()