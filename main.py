import math
import time
from Math.Geometry.GeometryPrimitives import *
from inputs.pathStorage import *
from SimulatedAlgorithms.localizer import *
from SimulatedAlgorithms.ramsete import *
from DataModels.robot import *
from DataModels.robot import GearTrain
from Visualizer.visualizer import *

time_step = 0.01
gear_train_l = GearTrain(0.75, 0.95, 0.374)
gear_train_r = GearTrain(0.75, 0.95, 0.374)
drive_wheel = DriveWheel(1000000,1.375)
left_motors = []
right_motors = []
for i in range (3):
    left_motors.append(Motor())
for i in range (3):
    right_motors.append(Motor())
horizontal_tracker = TrackerWheel(1.375,100000,0.07)
vertical_tracker = TrackerWheel(1,1000000,0.18)
inertial = IMU()
feed_forward = FeedForward(0.52, 1.155, 0.020)
drive_train = DriveTrain(gear_train_l,gear_train_r,drive_wheel,left_motors,right_motors,11.5, feed_forward)
robot = Robot(11.5,12.0,Pose(Vector2D(0.0,0.0),Angle(0.0)),7,drive_train,horizontal_tracker,vertical_tracker,inertial)
odom = TwoAxisTwoWheelOdom(horizontal_tracker,vertical_tracker,inertial)
localizer = SimpleLocalizer(odom)
dynamics = ChassisDynamics(robot, time_step)
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

while curr_time < path_one_full.get_duration()+5 :
    path_follower.update(curr_time)
    dynamics.update()
    visualizer.draw()
    curr_time+= time_step
visualizer.show()