from Math.Geometry.GeometryPrimitives import *
from DataModels.robot import *
import math
class TwoAxisTwoWheelOdom:
    def __init__(self, horizontal_tracker: TrackerWheel, vertical_tracker: TrackerWheel, inertial : IMU):
        self.horizontal_tracker = horizontal_tracker
        self.vertical_tracker = vertical_tracker
        self.prev_horizontal_tracker_dis = 0.0
        self.prev_vertical_tracker_dis = 0.0
        self.prev_heading = Angle(0.0)
        self.inertial = inertial

    def update(self):
        horizontal_distance = self.horizontal_tracker.get_position()
        vertical_distance = self.vertical_tracker.get_position()
        dv = vertical_distance - self.prev_vertical_tracker_dis
        dh = horizontal_distance - self.prev_horizontal_tracker_dis
        current_heading = self.inertial.get_heading()
        dtheta = (current_heading - self.prev_heading).normalize()

        self.prev_horizontal_tracker_dis = horizontal_distance
        self.prev_vertical_tracker_dis = vertical_distance

        self.prev_heading = current_heading
        return Pose(Vector2D(dh,dv),dtheta)
class SimpleLocalizer:
    def __init__(self,odom_model : TwoAxisTwoWheelOdom):
        self.pose = Pose(Vector2D(0.0,0.0), Angle(0.0))
        self.odom_model = odom_model
    def update(self):
        delta_pose = self.odom_model.update()
        mid_heading = (self.pose.heading + delta_pose.heading * 0.5).normalize()
        g_delta_pose = delta_pose.point.rotate(mid_heading.radians)
        self.pose.point += g_delta_pose
        self.pose.heading = (self.pose.heading + delta_pose.heading).normalize()
    def get_pose(self):
        return self.pose