import math
import numpy as np

from Math.Geometry.GeometryPrimitives import Vector2D, Pose, Angle
from inputs.pathStorage import Parser
from SimulatedAlgorithms.localizer import TwoAxisTwoWheelOdom, SimpleLocalizer
from SimulatedAlgorithms.ramsete import Ramsete, FeedForward
from DataModels.robot import (
    Robot,
    DriveTrain,
    Motor,
    GearTrain,
    DriveWheel,
    TrackerWheel,
    IMU,
    ChassisDynamics,
)
from DataModels.frictionForces import FrictionForce


class RobotEnv:
    def __init__(self):
        self.time_step = 0.01
        self.time = 0.0

        # Maximum physical residual.
        self.max_residual_x = 2.0
        self.max_residual_y = 2.0
        self.max_residual_heading = math.radians(10.0)

        # Controls how quickly residual authority increases.
        self.residual_tau = 1.0

        self.path_file = "inputs/pathOne.txt"

        self._build_simulation()

    def _build_simulation(self):
        """Create a fresh robot simulation."""

        self.gear_train_l = GearTrain(
            0.75,
            1.0,
            0.374,
        )

        self.gear_train_r = GearTrain(
            0.75,
            1.0,
            0.374,
        )

        self.drive_wheel = DriveWheel(
            1000000,
            1.625,
        )

        self.left_motors = [
            Motor()
            for _ in range(3)
        ]

        self.right_motors = [
            Motor()
            for _ in range(3)
        ]

        self.horizontal_tracker = TrackerWheel(1.375)
        self.vertical_tracker = TrackerWheel(1.0)

        self.inertial = IMU()

        self.feed_forward = FeedForward(
            0.52,
            1.155,
            0.02,
        )

        self.lateral_friction_force = FrictionForce(
            0.09,
            0.075,
        )

        self.forward_friction_force = FrictionForce(
            1.0,
            0.95,
        )

        self.rotational_friction_force = FrictionForce(
            0.95,
            0.94,
        )

        self.drive_train = DriveTrain(
            self.gear_train_l,
            self.gear_train_r,
            self.drive_wheel,
            self.left_motors,
            self.right_motors,
            11.5,
            self.feed_forward,
        )

        self.robot = Robot(
            11.5,
            12.0,
            Pose(
                Vector2D(0.0, 0.0),
                Angle(0.0),
            ),
            6.75,
            self.drive_train,
            self.horizontal_tracker,
            self.vertical_tracker,
            self.inertial,
        )

        self.odom = TwoAxisTwoWheelOdom(
            self.horizontal_tracker,
            self.vertical_tracker,
            self.inertial,
        )

        self.localizer = SimpleLocalizer(
            self.odom
        )

        self.dynamics = ChassisDynamics(
            self.robot,
            self.time_step,
            self.lateral_friction_force,
            self.rotational_friction_force,
            self.forward_friction_force,
        )

        self.text_parser = Parser(
            self.path_file
        )

        self.path_one = self.text_parser.parse()
        self.path_one_full = self.text_parser.parse(True)

        self.path_follower = Ramsete(
            self.localizer,
            self.robot,
            self.path_one_full,
            0.028,
            0.23,
        )

        self.episode_duration = (
            self.path_one_full.get_duration()
        )

    def reset(self):
        """Reset the environment to the beginning of an episode."""

        self.time = 0.0

        self._build_simulation()

        return self.get_actor_state()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_actor_state(self) -> np.ndarray:
        """
        State available to the actor.

        [estimated_x_error,
         estimated_y_error,
         estimated_heading_error,
         time]
        """

        error = self.get_estimated_pose_error()

        return np.array(
            [
                error.point.x,
                error.point.y,
                error.heading.radians,
                self.time,
            ],
            dtype=np.float32,
        )

    def get_privileged_state(self) -> np.ndarray:
        """
        State available only to the critic.

        [true_x_error,
         true_y_error,
         true_heading_error,
         true_vx,
         true_vy,
         true_angular_velocity]
        """

        error = self.get_true_pose_error()

        return np.array(
            [
                error.point.x,
                error.point.y,
                error.heading.radians,
                self.dynamics.local_vel.x,
                self.dynamics.local_vel.y,
                self.dynamics.angular_vel,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Pose error
    # ------------------------------------------------------------------

    def get_estimated_pose_error(self) -> Pose:
        """
        Calculate target pose error using the estimated robot pose.
        """

        self.localizer.update()

        estimated_pose = self.localizer.get_pose()

        target = self.path_one_full.sample_by_time(
            self.time
        )

        return self._calculate_pose_error(
            target.pose,
            estimated_pose,
        )

    def get_true_pose_error(self) -> Pose:
        """
        Calculate target pose error using the true simulation pose.
        """

        target = self.path_one_full.sample_by_time(
            self.time
        )

        return self._calculate_pose_error(
            target.pose,
            self.robot.pose,
        )

    def _calculate_pose_error(
        self,
        target: Pose,
        current: Pose,
    ) -> Pose:

        global_error = (
            target.point - current.point
        )

        local_error = global_error.rotate(
            -current.heading.radians
        )

        heading_error = (
            target.heading - current.heading
        ).normalize()

        return Pose(
            local_error,
            heading_error,
        )

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    def get_residual_scale(self) -> np.ndarray:
        """
        Return the maximum physical residual at the current time.
        """

        growth = (
            1.0
            - math.exp(
                -self.time / self.residual_tau
            )
        )

        return np.array(
            [
                self.max_residual_x * growth,
                self.max_residual_y * growth,
                self.max_residual_heading * growth,
            ],
            dtype=np.float32,
        )

    def action_to_residual(
        self,
        action: np.ndarray,
    ) -> Pose:
        """
        Convert normalized actor output [-1, 1] into a
        physical residual expressed in the robot frame.
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        residual = (
            action
            * self.get_residual_scale()
        )

        return Pose(
            Vector2D(
                residual[0],
                residual[1],
            ),
            Angle(
                residual[2]
            ),
        )

    def residual_to_global(
        self,
        residual: Pose,
    ) -> Pose:
        """
        Convert the residual translation from robot frame
        to global frame.

        Heading residual remains an angular offset.
        """

        global_translation = (
            residual.point.rotate(
                self.robot.pose.heading.radians
            )
        )

        return Pose(
            global_translation,
            residual.heading,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: np.ndarray,
    ):
        """
        Apply one residual action and advance the simulation.
        """

        residual = self.action_to_residual(
            action
        )

        global_residual = (
            self.residual_to_global(
                residual
            )
        )

        # Run Ramsete with the residual.
        self.path_follower.update(
            self.time,
            True,
            global_residual,
        )

        # Advance physics.
        self.dynamics.update()

        # Advance simulation time.
        self.time += self.time_step

        # Check termination.
        done = (
            self.time
            >= self.episode_duration
        )

        # Get next observations.
        next_actor_state = (
            self.get_actor_state()
        )

        next_privileged_state = (
            self.get_privileged_state()
        )

        # Calculate reward from TRUE state.
        true_error = (
            self.get_true_pose_error()
        )

        position_error = math.sqrt(
            true_error.point.x ** 2
            + true_error.point.y ** 2
        )

        heading_error = abs(
            true_error.heading.radians
        )

        reward = (
            -position_error
            -2.0 * heading_error
        )

        info = {
            "position_error": position_error,
            "heading_error": heading_error,
            "time": self.time,
        }

        return (
            next_actor_state,
            next_privileged_state,
            reward,
            done,
            info,
        )