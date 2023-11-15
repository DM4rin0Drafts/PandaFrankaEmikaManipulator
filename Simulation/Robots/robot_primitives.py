from Simulation.utils.pybullet_tools.pybullet_utils import all_joints, change_constraint, change_dynamics_gripper, \
    get_joints, link_collision, motor_control, motor_control_individual, object_collision, pairwise_link_collision, \
    all_links, get_limits_of_joint_info, get_joint_position, draw_line, get_all_joint_config, step_simulation, \
    remove_constraint, remove_debug_item, set_joint_positions, get_all_joint_position, check_limits, \
    create_constraint, disconnect, get_all_joint_velocity, get_current_motor_joint_state

from Simulation.Robots.robot_utils import CoordinateSystem, calculate_distance, transform_reference_systems, \
    check_costum_limits_trajectory, random_arm_config
from Simulation.Tests.ex4_tests import test_coefficients, test_q_t, test_qd_t, test_qdd_t
from Simulation.utils.utils import ReplayBuffer, check_custom_limits
from Simulation.Robots.never_collision import PANDA_NEVER_COLLISIONS
from Simulation.Tests.ex2_tests import test_transformation_matrix
from Simulation.utils.logging.runtime_handler import info_logger

from Tasks.ex4_tasks import compute_coefficient, q_t, qd_t, qdd_t
from Tasks.ex2_tasks import compute_transformation
from Tasks.ex3_tasks import rne

from scipy.spatial.transform import Rotation
from spatialmath.base import *
from spatialmath import SE3

import pybullet as p
import numpy as np
import time
import sys


useRealTimeSimulation = 0
useSimulation = 1
offset = 0.1

PI = np.pi
INF = np.Inf

base_limits = 0
CIRCULAR_LIMITS = -PI, PI


class Kinematic(object):
    def __init__(self, robot):
        """
        Object for the robot kinematic. Calculates the forward and backward kinematics

        Parameters
        ----------
            robot: RobotSetup, required
                Object dscription of the robot model with all important information
        """
        # super().__init__()
        self.robot = robot

    def compute_tcp_position(self, q, to_global=False):
        """
        Compute the tcp position of the end-effector with a given joint configuration

        Parameters
        ----------
            q: numpy, required
                Vector of the joint configuration as a flatten numpy array
            to_global: bool, optional
                Return end-effector position in local coordinate system or global coordinate system
        return
        ----------
            Returns [X, Y, Z] position as a flatten numpy array
        """
        t = self.robot.model.fkine(q).t

        if to_global:
            return transform_reference_systems(self.robot.base_position, t)
        else:
            return t

    def get_current_tcp_values(self, to_global=True):
        """
        Compute tcp position and the tcp orientation of the current position in the pybullet simulation.

        Parameters
        ----------
            to_global: bool, optional
                Return end-effector position in local coordinate system or global coordinate system
        return
        ----------
            Return tcp position and orientation as a flatten numpy array
        """
        joints = get_all_joint_position(self.robot.body)[:self.robot.dof]

        tcp = self.compute_tcp_position(joints, to_global)
        ori = self.compute_tcp_orientation(joints)

        return tcp, ori

    def compute_tcp_orientation(self, q):
        """
        Compute the tcp orientation of the end-effector with a given joint configuration and return it as euler angles.

        Parameters
        ----------
            q: numpy, required
                Vector of the joint configuration as a flatten numpy array
        return
        ----------
            Return the tcp orientation as a flatten numpy array
        """
        return np.array((Rotation.from_matrix(self.robot.model.fkine(q).R).as_euler('xyz', degrees=False)))

    @staticmethod
    def compute_transformation_matrix(target_position, target_orientation):
        """
        Compute the transformation matrix for the target end-effector position as a SE3 object

        Parameters
        ----------
            target_position: list/numpy, required
                Vector of the target position. Can be a 3-dimensional list or a flatten 3-dimensional array
            target_orientation: list/numpy, required
                Vector of euler angles. Can be a 3-dimensional list or a flatten 3-dimensional array
        return
        ----------
            Return a 4x4 transformation matrix of a SE3 object
        """
        transformation = SE3(target_position[0], target_position[1], target_position[2]) * \
                         SE3.Rz(target_orientation[2]) * SE3.Ry(target_orientation[1]) * SE3.Rx(target_orientation[0])
        return transformation

    def ikine_rtb(self, T, ilimit, rlimit, tol, search, slimit, q0=None):
        """
        Compute the inverse kinematic with the robotics-toolbox-library.

        Parameters
        ----------
            T: SE3, required
                The desired end-effector pose or pose trajectory
            ilimit: int, required
                maximum number of iterations (default 500)
            rlimit: int, required
                maximum number of consecutive step rejections (default 100)
            tol: float, required
                final error tolerance (default 1e-10)
            search: bool, required
                if True, then global search, else local search
            slimit: int, required
                maximum number of search attempts
            q0: numpy, optional
                initial joint configuration (default all zeros)

        return
        ----------

        """
        if isinstance(q0, np.ndarray):
            if search:
                # when initial joint config is given
                ik = self.robot.model.ikine_LM(T, ilimit=ilimit, q0=q0, rlimit=rlimit, tol=tol, search=search,
                                               slimit=slimit)
            else:
                ik = self.robot.model.ikine_LM(T, ilimit=ilimit, q0=q0, rlimit=rlimit, tol=tol, slimit=slimit)
        else:
            if search:
                ik = self.robot.model.ikine_LM(T, ilimit=ilimit, rlimit=rlimit, tol=tol, search=search, slimit=slimit)
            else:
                ik = self.robot.model.ikine_LM(T, ilimit=ilimit, rlimit=rlimit, tol=tol, slimit=slimit)

        return ik.q


class InverseKinematic(Kinematic):
    def __init__(self, robot, all_bodies, num_attempts=10, collision=False, ilimit=500, rlimit=100,
                 tol=1e-10, search=True, slimit=100):
        """
        Calculates inverse kinematics. Class can do collision check in pybullet simulation and searching
        all joint configuration of a given trajectory

        Parameters
        ----------
            robot: RobotSetup, required
                Object description of the robot model with all important information
            all_bodies: list, required
                list of all bodies in the simulation
            num_attempts: int, optional
                maximum number of attempts for searching an inverse kinematic
            collision: bool, optional
                do collision check in simulation
            ilimit: int, optional
                maximum number of iterations (default 500)
            rlimit: int, optional
                maximum number of consecutive step rejections (default 100)
            tol: float, optional
                final error tolerance (default 1e-10)
            search: bool, optional
                if True, then global search, else local search
            slimit: int, optional
                maximum number of search attempts
        return
        ----------

        """
        super().__init__(robot)
        self.all_bodies = all_bodies
        self.movable_joints = self.robot.movable_joints  # ignore finger joints
        self.num_attempts = num_attempts
        self.collision = collision
        self.ilimit = ilimit
        self.rlimit = rlimit
        self.tol = tol
        self.search = search
        self.slimit = slimit

        self.history = []

        # Test programming task if correct implemented
        test_result = test_transformation_matrix(compute_transformation)
        if test_result:
            self.test_successful = True
        else:
            self.test_successful = False

    def collision_check(self, q):
        """
        Check collision with a specific joint configuration.

        Parameters
        ----------
            q: list/numpy, required
                Vector of the joint configuration as a flatten numpy array
        return
        ----------
            Return boolean for collision. If true, robot has a collision with a different body in simulation.
        """
        # set joint configuration in simulation
        set_joint_positions(self.robot.body, self.movable_joints, q)

        no_collision = not pairwise_link_collision(self.robot.body,
                                                   all_links(self.robot.body),
                                                   self.robot.body,
                                                   all_links(self.robot.body),
                                                   kwargs=PANDA_NEVER_COLLISIONS)
        return no_collision and (not any(
            object_collision(self.robot.body, body2) for body2 in self.all_bodies))

    def search_ik(self, input_tuple, search_type=True, q0=None, local_search=False):
        """
        Search inverse kinematic

        Parameters
        ----------
            input_tuple: tuple, required
                Requires a 3D flatten numpy vector of the target tcp position in global coordiante system
                Requires a 3d flatten numpy vector of the tcp orientation (euler angles)
                Requires a target body id for ignoring a collision in the pybullet simulation (int or None)
            search_type: bool, optional
                Search new inverse kinematic joint position if tcp position is already in tolerance, else do nothing
            q0: numpy, optional
                Start position for searching a new inverse kinematic
            local_search: bool, optional
                if True, then local inverse kinematic search, else global inverse kinematic search
        return
        ----------
            Return a list of the found tcp position, the joint position and a boolean value if a configuration is found
        """
        target_position, target_orientation, target_body = input_tuple
        self.all_bodies = list(set(self.all_bodies) - {target_body})

        # Code for Exercise 2, check if tcp position is smaller then tolerance
        if not search_type:
            # if false, already near target, return tcp and current joint_config
            if isinstance(q0, np.ndarray):
                curr_position = self.compute_tcp_position(q0, True)
                curr_orientation = self.compute_tcp_orientation(q0)
            else:
                q = get_all_joint_position(self.robot.body)[:self.robot.dof]

                curr_position = self.compute_tcp_position(q, True)
                curr_orientation = self.compute_tcp_orientation(q)

            if calculate_distance(curr_position, target_position) < self.tol and \
                    calculate_distance(curr_orientation, target_orientation) < self.tol:
                return [curr_position, get_all_joint_position(self.robot.body)[:7], True]

        # Test if Exercise 2 task is correct implemented and build transformation_matrix
        if self.test_successful:
            local_tcp = transform_reference_systems(self.robot.base_position, target_position, False)
            T = SE3(compute_transformation(local_tcp, target_orientation))
        else:
            # current position
            joint_positions = get_all_joint_position(self.robot.body)[:self.robot.dof]
            tcp = self.compute_tcp_position(joint_positions)
            return [tcp, joint_positions, False]

        for _ in range(self.num_attempts):
            if local_search:
                # local search
                joint_config = self.ikine_rtb(T, ilimit=self.ilimit, q0=q0, rlimit=self.rlimit,
                                              tol=self.tol, search=False, slimit=self.slimit)
            else:
                # GLOBAL SERACH, USING ALWAYS NEW RANDOM CONFIG
                joint_config = self.ikine_rtb(T, ilimit=self.ilimit, q0=q0, rlimit=self.rlimit,
                                              tol=self.tol, search=self.search, slimit=self.slimit)

            # compute tcp position and orientation for checking tcp position is in range
            found_tcp = self.compute_tcp_position(joint_config, True)
            found_ori = self.compute_tcp_orientation(joint_config)

            diff_tcp = calculate_distance(target_position, found_tcp)
            diff_ori = calculate_distance(target_orientation, found_ori)

            # check joint limits and distances for a found configuration and a default rtb configuration
            if isinstance(joint_config, np.ndarray) and not local_search:
                if check_custom_limits(joint_config, self.robot.joint_limits[:, 0][:self.robot.dof],
                                       self.robot.joint_limits[:, 1][:self.robot.dof]) \
                        and diff_tcp < self.tol and diff_ori < self.tol:
                    break
            else:
                if check_custom_limits(joint_config, self.robot.joint_limits[:, 0][:self.robot.dof],
                                       self.robot.joint_limits[:, 1][:self.robot.dof]) \
                        and diff_tcp < self.tol and diff_ori < self.tol:
                    break
        else:
            # No inverse kinematic found, set joint configuration to infinity
            joint_config = np.ones((self.robot.dof,)) * np.infty

        if self.collision:
            # Test collision in pybullet simulation
            if self.collision_check(joint_config) and not np.isinf(joint_config).all():  # check collision
                info_logger('GOOD ik CONFIGURATION FOUND\n\n')

                current_tcp = self.compute_tcp_position(joint_config)
                # reset-to-start-arm-position
                set_joint_positions(self.robot.body, self.movable_joints, joint_config)

                return [current_tcp, joint_config, True]
            else:
                # TODO not print do logging
                print("no configuration found - returning current joint position")
                joint_positions = get_all_joint_position(self.robot.body)[:self.robot.dof]
                tcp = self.compute_tcp_position(joint_positions)

                return [tcp, joint_positions, False]

        else:
            # if collision check is not needed. Check if a joint configuration is found
            if not np.isinf(joint_config).all():
                current_tcp = self.compute_tcp_position(joint_config, True)
                return [current_tcp, joint_config, True]
            else:
                print("no configuration found - returning current joint position")
                joint_positions = get_all_joint_position(self.robot.body)[:self.robot.dof]
                tcp = self.compute_tcp_position(joint_positions)

                return [tcp, joint_positions, False]

    def joint_space_search(self, tcp_position, tcp_orientation, attempts):
        """
        Method for searching a trajectory in the joint space.

        Parameters
        ----------
            tcp_position: numpy, required
                Vector of the target position. Can be a 3-dimensional list or a flatten 3-dimensional
            tcp_orientation: list/numpy, required
                Vector of euler angles. Can be a 3-dimensional list or a flatten 3-dimensional array
            attempts: int, required
                Attempts for trying to search a solution
        return
        ----------
            Return a boolean value for a found path
        """
        # if history is empty, use the current joint position, else use the last history joint position as a start
        # joint configuration q0
        config = None

        if not self.history:
            curr_joints = get_all_joint_position(self.robot.body)[:self.robot.dof]
            self.history.append([np.array(tcp_position[0]), np.array(curr_joints), True])
        else:
            last_config = self.history[-1]
            curr_joints = last_config[1].copy()
        
        for i in range(attempts):
            # add a random variance to the current joint position
            q0 = random_arm_config(self.robot, curr_joints, i, attempts)

            # search inverse kinematic with a local search
            config = self.search_ik((tcp_position[1], tcp_orientation[1], None), q0=q0, local_search=True)
            if config[2]:
                self.history.append(config)
                break
        else:
            # Case for not finding an inverse kinematic. Do global search
            for _ in range(attempts):
                config = self.search_ik((tcp_position[1], tcp_orientation[1], None))
                if config[2]:
                    self.history.append(config)
                    break

        if not config[2]:
            # no config found
            # TODO delete all new added elements in history 
            return False
        else:
            return True

    def cartesian_search(self, tcp_positions, tcp_orientations, attempts):
        """
        Method for searching a trajectory in the cartesian space.

        Parameters
        ----------
            tcp_positions: numpy, required
                Matrix of the target position. Is a 3-dimensional matrix with a shape of (n, 3)
            tcp_orientations: numpy, required
                Matrix of the target orientations. Is a 3-dimensional matrix with a shape of (n, 3)
            attempts: int, required
                Attempts for trying to search a solution
        return
        ----------
            Return a boolean value for a found path. Result is saved in history as a list
        """

        found_solutions = 0

        for idx, (tcp_target, tcp_orientation) in enumerate(zip(tcp_positions, tcp_orientations)):
            # take one tcp position and orientation and search an inverse kinematic
            for i in range(attempts):
                # if history is empty, use the current joint position, else use the last history joint position
                # as a start joint configuration q0
                if not self.history:
                    curr_joints = np.array(get_all_joint_position(self.robot.body)[:self.robot.dof])
                else:
                    last_config = self.history[-1]
                    curr_joints = last_config[1].copy()

                # add a random variance to the current joint position
                q0 = random_arm_config(self.robot, curr_joints, i, attempts)

                # search inverse kinematic with a local search
                config = self.search_ik((tcp_target, tcp_orientation, None), q0=q0, local_search=True)

                if config[2]:
                    self.history.append(config)
                    found_solutions += 1
                    break

            # check if a tcp position is found
            if idx + 1 != found_solutions:
                # TODO DO logging
                print("path not found")
                # TODO delete elements in history
                return False

        if found_solutions == len(tcp_positions):
            return True
        else:
            return False

    def search_ik_plan(self, input_tuple, mode=True, attempts=25):
        """
        Search inverse kinematic for a target positions and target orientation with multiple

        Parameters
        ----------
            input_tuple: tuple, required
                Gets list of tcp_positions and tcp_orientations (see cartesian_search or joint_space_search)
            mode: bool, optional
                Mode for space search. if true, do joint space search, else cartesian search
            attempts: int, optional
                Attempts for trying to search a solution
        return
        ----------
            Return boolean value for a found plan. Result is saved in history as a list
        """
        tcp_positions, tcp_orientations = input_tuple

        if mode:
            # joint space search
            # we are only using 3 attempts, because it takes to long for finding target position in long distances
            result = self.joint_space_search(tcp_positions, tcp_orientations, attempts=1)
        else:
            # cartesian search
            result = self.cartesian_search(tcp_positions, tcp_orientations, attempts)

        return result


class PolynomialTrajectory(object):
    def __init__(self, robot):
        """
        Class for computing a polynomial trajectory in the joint space.

        Parameters
        ----------
            robot: RobotSetup, required
                Object description of the robot model with all important information
        """
        self.robot = robot

    @staticmethod
    def vec_2_matrix(vec, n):
        """
        Converts a vector to a matrix

        Parameters
        ----------
            vec: numpy, required
                Flatten numpy array that should be n-times copy
            n: int, required
                Number of rows that should be copied
        return
        ----------
            Return a numpy matrix of a shape of (n, len(vec))
        """
        return np.tile(vec, (n, 1))

    @staticmethod
    def linear_interpolation(target, current, time_steps, step=60):
        """
        Build a straight line by using linear interpolation form one vector to another vector

        Parameters
        ----------
            target: numpy, required
                Vector of the target point as a flatten numpy array
            current: numpy, required
                Vector of the current point as a flatten numpy array
            time_steps: int, required
                Execution time form the current point to the target point in seconds
            step: int, optional
                step size of the pybullet execution time
        return
        ----------
            Return a (n, 3) numpy array for the linear interpolated
        """
        diff = target - current
        interval = diff / (time_steps * step - 1)
        targets = []
        for i in range((time_steps * step)):
            value = current + interval * i
            targets.append(value)

        return np.array(targets).copy()

    def compute_trajectory(self, tcp_joint_position, qd_tuple, qdd_tuple, t, steps=60):  # manchmal
        """
        Compute the joint position, joint velocity and joint acceleration for a trajectory.

        Parameters
        ----------
            tcp_joint_position: tuple, required
                Start and end tcp joint position as a numpy flatten array
            qd_tuple: tuple, required
                Start and end tcp joint velocity as a numpy flatten array
            qdd_tuple: tuple, required
                Start and end tcp joint acceleration as a numpy flatten array
            t: tuple, required
                Execution time range. Fist element has to be ALWAYS zero. Execution stop has to be an integer.
            steps: int, optional
                step size of the pybullet execution time
        return
        ----------
            Return joint position, joint velocity and joint acceleration as a numpy array as a shape of (n, dof)
        """
        ti = np.linspace(t[1], t[0], num=(t[1] - t[0]) * steps)
        ti = ti.reshape((len(ti), 1))
        a_0, a_1, a_2, a_3, a_4, a_5 = compute_coefficient(tcp_joint_position, qd_tuple, qdd_tuple, t)

        a_0 = self.vec_2_matrix(a_0, len(ti))
        a_1 = self.vec_2_matrix(a_1, len(ti))
        a_2 = self.vec_2_matrix(a_2, len(ti))
        a_3 = self.vec_2_matrix(a_3, len(ti))
        a_4 = self.vec_2_matrix(a_4, len(ti))
        a_5 = self.vec_2_matrix(a_5, len(ti))

        qt = q_t(a_0, a_1, a_2, a_3, a_4, a_5, ti)
        dqt = qd_t(a_1, a_2, a_3, a_4, a_5, ti)
        ddqt = qdd_t(a_2, a_3, a_4, a_5, ti)

        if not isinstance(qt, np.ndarray) or not isinstance(dqt, np.ndarray) or not isinstance(ddqt, np.ndarray):
            return None, None, None

        if not check_costum_limits_trajectory(qt, self.robot.joint_limits[:, 0][:self.robot.dof],
                                              self.robot.joint_limits[:, 1][:self.robot.dof]) \
                or not check_costum_limits_trajectory(dqt, -self.robot.max_velocity, self.robot.max_velocity) \
                or not check_costum_limits_trajectory(ddqt, -self.robot.max_acceleration,
                                                      self.robot.max_acceleration):
            # TODO logging here
            return None, None, None

        return qt[::-1], dqt[::-1], ddqt[::-1]


class PathPlanning(PolynomialTrajectory, InverseKinematic, CoordinateSystem):
    joint_planning_path = None
    q, dq, ddq = None, None, None
    tcp_position_planning = []

    def __init__(self, robot, all_bodies, movable_bodies=None, collision=False, steps=60, visualization=False):
        """
        Class for planning a trajectory in cartesian space and joint space.

        Parameters
        ----------
            robot: RobotSetup, required
                Object description of the robot model with all important information
            all_bodies: list, required
                List of all bodies in the simulation
            movable_bodies: list, optional
                List of all movable bodies in the simulation
            collision: bool, optional
                Boolean for collision checking in the pybullet simulation
            steps: int, optional
                Simulation step size
            visualization: bool, optional
                Boolean for visualize tcp position the pybullet simulation
        """
        PolynomialTrajectory.__init__(self, robot)
        InverseKinematic.__init__(self, robot, all_bodies)
        CoordinateSystem.__init__(self, robot.body)

        self.robot = robot
        self.all_bodies = all_bodies
        self.movable_bodies = movable_bodies
        self.collision = collision
        self.steps = steps
        self.visualization = visualization

        # load dynamic model
        self.dyn = Dynamic(self.robot)

        # save joint position, joint velocity and joint acceleration as a [n, dof] shape numpy array
        self.q = np.array(get_all_joint_position(self.robot.body)[:self.robot.dof])[None, :]
        self.qd = np.array(get_all_joint_velocity(self.robot.body)[:self.robot.dof])[None, :]
        self.qdd = np.array([0.0] * self.robot.dof)[None, :]

        # save tcp position in a list
        self.eef_positions = [self.compute_tcp_position(self.q, True)]
        self.draw_trajectory_lines = [] 
                            
        # test implementations tasks
        if test_coefficients(compute_coefficient) and test_q_t(q_t) \
                and test_qd_t(qd_t) and test_qdd_t(qdd_t):
            self.successful_test = True
        else:
            self.successful_test = False

    def draw_trajectory(self, q=None, local=True, color=(1, 0, 0), width=0.8):
        """
        Draws a given trajectory into the pybullet simulation

        Parameters
        ----------
            q: numpy, optional
                joint configuration of the trajectory
            local: bool, optional
                If input of trajectory is in local or global space
            color: tuple, required
                Tuple of RGB color representation
            width: float, optional
                Width of the line
        """
        if isinstance(q, np.ndarray):
            tcp = []
            for qt in q:
                if local:
                    tcp.append(self.compute_tcp_position(qt, True))
                else:
                    tcp.append(self.compute_tcp_position(qt))

            self.eef_positions = np.asarray(tcp)

        for i in range(1, len(self.eef_positions)):
            value = draw_line([self.eef_positions[i - 1], self.eef_positions[i]],
                              color, width)
            self.draw_trajectory_lines.append(value)

    def clear_trajectory_lines(self):
        """
        Clears and delete all lines in the pybullet simulation
        """
        for value in self.draw_trajectory_lines:
            remove_debug_item(value)

    def clear_variables(self):
        # save joint position, joint velocity and joint acceleration as a [n, dof] shape numpy array
        self.q = np.array(get_all_joint_position(self.robot.body)[:self.robot.dof])[None, :]
        self.qd = np.array(get_all_joint_velocity(self.robot.body)[:self.robot.dof])[None, :]
        self.qdd = np.array([0.0] * self.robot.dof)[None, :]

        # save tcp position in a list
        self.eef_positions = [self.compute_tcp_position(self.q, True)]
        self.draw_trajectory_lines = [] 

        # clear history 
        self.history = []

    def plan_execution_path(self, tcp_positions, tcp_orientations, execution_pattern, execution_time,
                            velocities=np.array([[0.0] * 7, [0.0] * 7]),
                            accelerations=np.array([[0.0] * 7, [0.0] * 7])):
        """
        Plans a trajectory in joint space and cartesian space.

        Parameters
        ----------
            tcp_positions: numpy, required
                Numpy array of tcp target position in a shape of (n, 3)
            tcp_orientations: numpy, required
                Numpy array of tcp target orientations in euler angles in a shape of (n, 3)
            execution_pattern: list , required
                Pattern for planning trajectory section in joint space (=1) or cartesian space(=0)
            execution_time: list, required
                Execution time of the duration in the trajectory sections as integers
            velocities: numpy, optional
                Start and end velocity of every trajectory section. Needs shape of (2, dof) and set
                on every section
            accelerations: numpy, optional
                Start and end acceleration of every trajectory section. Needs shape of (2, dof) and set
                on every section
        return
        ----------
            Return True, if a plan is found for a given path, else None
        """
        track_eef_position = []

        start_position_joint = get_all_joint_position(self.robot.body)[:self.robot.dof]
        start_position_tcp = self.compute_tcp_position(start_position_joint, True).reshape((1, 3))
        start_orientation = self.compute_tcp_orientation(start_position_joint).reshape((1, 3))

        if tcp_positions.ndim == 1:
            tcp_positions = tcp_positions[None, :]

        tcp_positions = np.vstack((start_position_tcp, tcp_positions))
        tcp_orientations = np.vstack((start_orientation, tcp_orientations))
        
        if self.successful_test:
            for idx, (pattern, t) in enumerate(zip(execution_pattern, execution_time)):
                search_config_tcp = np.vstack((tcp_positions[idx], tcp_positions[idx + 1]))
                search_config_orientation = np.vstack((tcp_orientations[idx], tcp_orientations[idx + 1]))

                if pattern:
                    # joint space search
                    found_plan = self.search_ik_plan((search_config_tcp, search_config_orientation))

                    if found_plan:
                        qt, qdt, qddt = self.compute_trajectory((self.history[-2][1], self.history[-1][1]),
                                                                (velocities[0], velocities[0]),
                                                                (accelerations[0], accelerations[0]), (0, t))
                    else:
                        return
                else:
                    # cartesian/task space search
                    cs_target_search = self.linear_interpolation(search_config_tcp[1],
                                                                 search_config_tcp[0],
                                                                 execution_time[idx])
                    cs_orientation_search = self.linear_interpolation(search_config_orientation[1],
                                                                      search_config_orientation[0],
                                                                      execution_time[idx])

                    found_plan = self.search_ik_plan((cs_target_search, cs_orientation_search), mode=False)
                    if found_plan:
                        joint_pos = [self.history[i][1] for i in range(len(self.history))]
                        qt = np.array(joint_pos[-t * self.steps:])
                        _, qdt, qddt = self.compute_trajectory((qt[-1], qt[0]), (velocities[0], velocities[0]),
                                                               (accelerations[0], accelerations[0]), (0, t))
                    else:
                        return

                if isinstance(qt, np.ndarray):
                    self.q = np.vstack((self.q, qt.copy()))
                    self.qd = np.vstack((self.qd, np.array(qdt).copy()))
                    self.qdd = np.vstack((self.qdd, np.array(qddt).copy()))
                else:
                    return

            for q_result in self.q:
                track_eef_position.append(self.compute_tcp_position(q_result, True))

            if self.visualization:
                self.eef_positions = np.asarray(track_eef_position)
                self.draw_trajectory()

            return True
        else:
            return


# see bug report (13. sept. 21) maybe newer version will fix it
# https://github.com/petercorke/robotics-toolbox-python/issues/255
class Dynamic(object):
    def __init__(self, robot, gravity=None, brake_task=False):
        """
        Class for calculating the dynamics of a model

        Parameters
        ----------
            robot: object, required
                Object description of the robot model with all important information
            gravity: list/numpy, optional
                Set individual gravity to dynamic model, else loading normal earth gravity
            brake_task: bool, optional
                Boolean for exercise 5. Set all joints > 1 to zero. These joint are now connected.
        """
        self.robot = robot
        self.dof = robot.model.n

        # rotation matrix
        self.R = [None for _ in range(self.dof)]
        # angular velocity
        self.w = [None for _ in range(self.dof)]
        # angular acceleration
        self.wd = [None for _ in range(self.dof)]

        self.v = [None for _ in range(self.dof)]
        self.vd = [None for _ in range(self.dof)]
        self.vdc = [None for _ in range(self.dof)]

        self.F = [None for _ in range(self.dof)]
        self.N = [None for _ in range(self.dof)]

        self.f = [None for _ in range(self.dof + 1)]
        self.n = [None for _ in range(self.dof + 1)]

        # inertia matrix
        self.I = [None for _ in range(self.dof)]
        # joint moition subspace
        self.e = [None for _ in range(self.dof)]
        self.t = [None for _ in range(self.dof)]

        # G = gear ratio, Jm = , B = , Tc =
        self.G = [None for _ in range(self.dof)]
        self.Jm = [None for _ in range(self.dof)]
        self.B = [None for _ in range(self.dof)]
        self.Tc = [None for _ in range(self.dof)]

        # self.a = [None for _ in range(self.dof)]

        # center of mass of the joints
        self.center_of_mass = [None for _ in range(self.dof)]
        # masses of the joints
        self.masses = [None for _ in range(self.dof)]

        # set gravity
        if gravity is None:
            self.a_grav = -self.robot.model.gravity
        else:
            self.a_grav = -gravity

        # load model values into list
        for index, link in enumerate(self.robot.model.links[1:self.dof + 1]):
            self.I[index] = np.array(link.I)
            self.center_of_mass[index] = np.array(link.r)
            self.G[index] = link.G
            self.Jm[index] = link.Jm
            self.B[index] = link.B
            self.Tc[index] = link.Tc

            if index == 0:
                # in rtb library matlab, there is a small differences in matlab by using the panda robot
                # needs to be set to zero. 
                self.center_of_mass[index][2] = 0.0
            self.e[index] = np.array(link.v.s, dtype=np.float64)

            if brake_task:
                if index > 0:
                    self.masses[index] = np.array([0])
                else:
                    self.masses[index] = np.array([link.m])
            else:
                self.masses[index] = np.array([link.m])

    def friction(self, qd, j):
        """
        Calculate friction

        Parameters
        ----------
            qd: numpy, required
                Joint velocity as a flatten numpy array
            j: int, required
                Index of the joints
        return
        ----------
            Return frictions as a numpy array as a flatten numpy array (dof)
        """
        tau = self.B[j] * abs(self.G[j]) * qd

        # Coulomb friction
        if qd > 0:
            tau = tau + self.Tc[j][0]
        elif qd < 0:
            tau = tau + self.Tc[j][1]

        tau = -abs(self.G[j]) * tau

        return tau

    def forward_recursion(self, qd, qdd, gravity=None):
        """
        Computes the forward recursion

        Parameters
        ----------
            qd: numpy, required
                joint velocity as a flatten numpy array
            qdd: numpy, required
                joint acceleration as a flatten numpy array
            gravity: list, optional
                Set individual gravity to dynamics, else loading normal earth gravity
        return
        ----------
            Return the center of gravity attacked on torque N and the center of gravity of the joint F
            as a 3x3 numpy array
        """
        F = [None for _ in range(self.dof)]
        N = [None for _ in range(self.dof)]

        for j in range(0, self.dof):
            if j == 0:
                wi = np.zeros(3)
                dwi = np.zeros(3)

                if gravity is None:
                    dvi = np.array(self.a_grav)
                else:
                    dvi = np.array(gravity)
            else:
                wi = self.w[j - 1]
                dwi = self.wd[j - 1]

                dvi = self.vd[j - 1]

            self.w[j] = self.R[j] @ wi + qd[j] * self.e[j][3:]
            self.wd[j] = self.R[j] @ dwi + np.cross(self.R[j] @ wi, qd[j] * self.e[j][3:]) + qdd[j] * self.e[j][3:]

            self.vd[j] = self.R[j] @ (np.cross(dwi, self.t[j]) + np.cross(wi, np.cross(wi, self.t[j])) + dvi)
            self.vdc[j] = np.cross(self.wd[j], self.center_of_mass[j]) + \
                          np.cross(self.w[j], np.cross(self.w[j], self.center_of_mass[j])) + self.vd[j]

            N[j] = self.I[j] @ self.wd[j] + np.cross(self.w[j], (self.I[j] @ self.w[j]))
            F[j] = self.masses[j] * self.vdc[j]

        return N, F

    def backward_recursion(self, qd, qdd, N, F):
        """
        Compute the forecs for all joints

        Parameters
        ----------
            qd: numpy, required
                joint velocity as a flatten numpy array
            qdd: numpy, required
                joint acceleration as a flatten numpy array
            N: numpy, required
                Center of gravity attacked on torque as list full of 3x3 numpy array
            F: numpy, required
                Center of gravity of the joint as a list full of 3x3 numpy array
        return
        ----------
            Return forces as a flatten numpy array
        """
        tau = np.zeros(qd.shape)

        for j in reversed(range(0, self.dof)):
            if j == self.dof - 1:
                self.f[j + 1] = np.zeros(3)
                self.n[j + 1] = np.zeros(3)
                R = np.eye(3)
                t = np.zeros(3)
            else:
                R = self.R[j + 1]
                t = self.t[j + 1]

            self.f[j] = (R.T @ self.f[j + 1][..., np.newaxis]).flatten() + F[j]
            self.n[j] = N[j] + (R.T @ self.n[j + 1]) + np.cross(self.center_of_mass[j], F[j]) \
                        + np.cross(t, (R.T @ self.f[j + 1]))

            tau[j] = self.e[j][3:].T @ self.n[j] + self.G[j] ** 2 + self.Jm[j] * qdd[j] - self.friction(qd[j], j)

        return tau


class Grasp(object):
    def __init__(self, robot, target_object=None, kp=np.array([1., 1.]), kv=np.array([.3, .3]),
                 control_mode=p.POSITION_CONTROL):
        """
        Class for grasping an object in the pybullet simulation. Depends on the URDF file.

        Parameters
        ----------
            robot: object, required
                Object description of the robot model with all important information
            target_object: int, optional
                ID of the pybullet object
            kp: numpy, optional
                Positioning gain as a numpy flattan array as a shape of (number of fingers,)
            kv: numpy, optional
                Velocity gain as a numpy flattan array as a shape of (number of fingers,)
            control_mode: int, optional
                Pybullet control mode like POSITION_CONTROL, VELOCITY_CONTROL and TORQUE_CONTROL
        """
        self.robot = robot
        self.target = target_object
        self.kp = kp
        self.kv = kv
        self.control_mode = control_mode
        self.gripper_limits = np.array(
            [get_limits_of_joint_info(self.robot.body, joint) for joint in
             self.robot.movable_gripper])

        self.cid = create_constraint(self.robot.body)
        change_constraint(self.cid)
        change_dynamics_gripper(self.robot.body, self.robot.movable_gripper)

    def motion_control_grasp(self):
        """
        Grasp object in pybullet simulation with motion control
        """
        if self.target is not None:
            current_position = np.array(
                [get_joint_position(self.robot.body, joint) for joint in self.robot.movable_gripper])
            while True:
                if object_collision(self.robot.body, self.target):
                    break

                elif (current_position <= self.gripper_limits[:, 0]).all():
                    # gripper is on minimum limits
                    break
                else:
                    # close gripper
                    current_position = current_position - 0.001
                    motor_control_individual(self.robot.body, self.robot.movable_gripper,
                                             current_position, self.control_mode,
                                             self.kp, self.kv)

            if not (current_position <= self.gripper_limits[:, 0]).all():
                # grasp the object a bit tighter
                current_position = current_position - 0.005
                motor_control_individual(self.robot.body,
                                         self.robot.movable_gripper,
                                         current_position,
                                         self.control_mode,
                                         self.kp,
                                         self.kv)

    def motion_control_open_gripper(self):
        """
        open gripper in pybullet simulation with motion control
        """
        if self.target is not None:
            while True:
                current_position = np.array(
                    [get_joint_position(self.robot.body, joint) for joint in
                     self.robot.movable_gripper])
                if (current_position >= self.gripper_limits[:, -1]).all():
                    break
                else:
                    current_position = current_position + 0.001
                    motor_control_individual(self.robot.body,
                                             self.robot.movable_gripper,
                                             current_position,
                                             self.control_mode,
                                             self.kp,
                                             self.kv)

            remove_constraint(self.cid)


class MotionControl(Dynamic, Grasp):
    def __init__(self, robot, trajectory, all_bodies, check_collision, target=None,
                 execution_times=None, control_mode=p.POSITION_CONTROL, kp=np.array([1.0] * 7),
                 kv=np.array([.3] * 7), dt=1. / 240., steps=60, buffer_size=10):
        """
        Class for following a trajectory in the pybullet simulation and control model in simulation.

        Parameters
        ----------
            robot: RobotSetup, required
                Object description of the robot model with all important information
            trajectory: tuple, required
                Tuple of joint position, joint velocity and joint acceleration as a (n, dof)
                numpy array
            all_bodies: list, required
                list of all bodies in the simulation
            check_collision: bool, required
                Boolean for checking collision in pybullet simulation
            target: int, optional
                Pybullet object ID that should be grasp in the simulation
            execution_times:
                Execution time for grasping and dropping an object in the pybullet simulation
            control_mode:
                Pybullet control mode like POSITION_CONTROL, VELOCITY_CONTROL and TORQUE_CONTROL
            kp: numpy, optional
                Positioning gain as a numpy flattan array as a shape of (dof,)
            kv: numpy, optional
                Velocity gain as a numpy flattan array as a shape of (dof,)
            dt: float, required
                Sleeping time in the pybullet simulation for the next iteration in pybullet
            buffer_size: int, optional
                Size of maximum values that should be tracked and saved (i.e. position, velocity,
                acceleration and forces)
        """
        Dynamic.__init__(self, robot)
        Grasp.__init__(self, robot, target)

        # required shape of (n, dof)
        self.q_pos_desired, self.q_vel_desired, self.q_acc_desired = trajectory

        self.execution_times = execution_times
        self.joint_states = get_all_joint_config(self.robot.body)
        self.control_mode = control_mode
        self.all_bodies = all_bodies
        self.check_collision = check_collision

        self.kp = kp
        self.kv = kv
        self.dt = dt
        self.steps = steps

        self.track_position = ReplayBuffer(buffer_size)
        self.track_velocity = ReplayBuffer(buffer_size)
        self.track_acceleration = ReplayBuffer(buffer_size)
        self.track_forces = ReplayBuffer(buffer_size)

        # time-step for dynamic exericse task
        self.t = 0

    def joint_tracking(self):
        """
        Method for tracking position and velocity and add it to the queue
        """
        max_velocity = self.robot.max_velocity

        curr_pos, curr_vel, _ = get_current_motor_joint_state(self.robot.body)
        check_limits(self.robot.movable_joints, curr_vel, max_velocity)

        # change shape and convert it to numpy array
        self.track_position.add(np.array(curr_pos).reshape((1, len(curr_pos))))
        self.track_velocity.add(np.array(curr_vel).reshape((1, len(curr_vel))))

    def check_motion_collision(self):
        """
        Check if a collision appears in a trajectory
        """
        joints = all_joints(self.robot.body)

        for body2 in self.all_bodies:
            for joint in joints:
                result = link_collision(self.robot.body, joint, body2)

                # ignore base collision
                if result and joint != 0:
                    print(body2, joint)
                    print("ERROR: COLLISION")
                    time.sleep(15)
                    disconnect()
                    sys.exit()

    def execute_motion_control(self, input_tuple=None, tracking=True):
        """
        Execution a trajectory in the pybullet simulation

        Parameters
        ----------
            input_tuple: tuple, optional
                Tuple with a size of 2. Get row index of the grasp and drop position in the trajectory
            tracking: bool, optional
                Boolean state for tracking joint position, joint velocity, joint acceleration and joint forces
        """
        grasp, drop = None, None

        if input_tuple is not None:
            grasp_index, drop_index = input_tuple
        else:
            grasp_index, drop_index = None, None

        # get grasp and drop index in the list of the trajectory
        if grasp_index is not None:
            grasp = sum(self.execution_times[:grasp_index]) * self.steps
        if drop_index is not None:
            drop = sum(self.execution_times[:drop_index]) * self.steps
        dyn = Dynamic(self.robot)

        if isinstance(self.q_pos_desired, np.ndarray) and isinstance(
                self.q_vel_desired, np.ndarray) and isinstance(self.q_acc_desired, np.ndarray):

            for idx, (t_pos, t_vel, t_acc) in enumerate(zip(self.q_pos_desired, self.q_vel_desired,
                                                            self.q_acc_desired)):
                # execute motion in pybullet simulation
                motor_control(self.robot.body, self.robot.movable_joints, self.control_mode, t_pos, t_vel,
                              position_gain=self.kp, velocity_gain=self.kv)

                step_simulation(sleep_time=self.dt)

                # track forces, position, velocity and acceleration and calculate it
                if tracking:
                    forces = rne(dyn, t_pos, t_vel, t_acc).reshape((1, self.robot.dof))

                    self.track_position.add(t_pos.reshape((1, self.robot.dof)))
                    self.track_velocity.add(t_vel.reshape((1, self.robot.dof)))
                    self.track_acceleration.add(t_acc.reshape((1, self.robot.dof)))
                    self.track_forces.add(forces)

                # Grasping and drop object in pybullet simulation
                if grasp_index is not None:
                    if idx == grasp:
                        self.motion_control_grasp()
                if drop_index is not None:
                    if idx == drop:
                        self.motion_control_open_gripper()

                # check collision in pybullet simulation
                if self.check_collision:
                    self.check_motion_collision()

    def execute_dynamic_motion_control(self, q_t, qd_t, qdd_t, tau):
        """
        Execute dynamic motion control in pybullet simulation

        Parameters
        ----------
            q_t: numpy, required
                joint position step as a flatten numpy array with a shape of (dof,)
            qd_t: numpy, required
                joint velocity step as a flatten numpy array with a shape of (dof,)
            qdd_t: numpy, required
                joint acceleration step as a flatten numpy array with a shape of (dof,)
            tau: numpy, required
                forces step as a flatten numpy array with a shape of (dof,)
        """
        # track all 30 time steps in the simulation
        if self.t % 30 == 0:
            self.joint_tracking()

            # compute forces and acceleration and save it in queue
            self.track_forces.add(
                np.array(tau).reshape((1, len(self.robot.movable_joints))))
            self.track_acceleration.add(
                np.asarray(qdd_t).reshape((1, len(qdd_t))))

        # execute motion in pybullet simulation
        motor_control(self.robot.body, self.robot.movable_joints,
                      self.control_mode, q=q_t, dq=qd_t,
                      position_gain=self.kp, velocity_gain=self.kv)
        step_simulation(sleep_time=self.dt)

        self.t += 1
