from Simulation.utils.pybullet_tools.pybullet_utils import all_joints, draw_text, euler_from_quaternion, \
    get_all_joint_position, joint_from_name, matrix_from_quaternion, quaternion_from_matrix, \
    quaternion_from_euler, remove_debug_item, set_joint_position, get_limits_of_joint_info, \
    get_fkine_position, get_fkine_orientation, draw_line, get_all_joint_limits
from Simulation.utils.utils import check_custom_limits
from Simulation.utils.utils import MultiThreading
import numpy as np

# Predefined configurations
START_POSITION = [0.0, 0.3993, 0.08, -1.6552, 0.1761, 2.6486, 0.0]
QR = [0., -0.3, 0., -2.2, 0., 2., 0.78539816]
COLLISION_TEST = [0.0, 1.0766, 0.0, -2.0477, -0.0800, 3.2318, 0.5603]
COLLISION_ROBOT_TEST = [0.0480, -1.1005, 0.2081, -3.0718, 0.3362, 0.5657, 0.9764]
PANDA_ACCELERATION = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
PANDA_VELOCITY = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

INITIAL_GRASP_POSITIONS = {
    "start": START_POSITION,
    "collision_test": COLLISION_TEST,
    "collision_robot_test": COLLISION_ROBOT_TEST,
    "qr": QR
}

# Names of the joints and finger in the robot, you can find it in the urdf-file
# or use ROS moveit
ROBOT_GROUPS = {
    'arm': ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6',
            'panda_joint7'],
    'gripper': ['panda_finger_joint1', 'panda_finger_joint2']
}

ROBOT_GROUPS_ID = {
    'arm': [i for i in range(7)],
    'gripper': [9, 10]
}

# Attention: work with ROBOT_GROUPS, values are in radians, NOT needed because of get_limits_of_joint_info()
# in Simulation/utils/pybullet_tools/pybullet_utils
ROBOT_LIMITS_ARM = {
    "panda_joint1": [-2.8973, 2.8973],
    "panda_joint2": [-1.7628, 1.7626],
    "panda_joint3": [-2.8973, 2.8973],
    "panda_joint4": [-3.0718, 0.0],
    "panda_joint5": [-2.8973, 2.8973],
    "panda_joint6": [0.0, 3.7525],
    "panda_joint7": [-2.8973, 2.8973],
}

ROBOT_LIMITS_GRIPPER = {
    "panda_finger_joint1": [0.0, 0.04],
    "panda_finger_joint2": [0.0, 0.04]
}

ROBOT_EEF_INDEX = {
    "panda": 11
}


def get_initial_grasp_type(type_of_grasp=None):
    """
    Load predefined joint position from a dictionary.

    Parameters
    ----------
        type_of_grasp: str, optional
            Key name of the dictionary.

    return
    ----------
        Return joint configuration of the type of grasp as a list
    """
    if type_of_grasp is None:
        return QR
    else:
        return INITIAL_GRASP_POSITIONS[type_of_grasp]


def get_joint_from_model(body, limbs):
    """
    Get ID number from a given robot limb as a tuple.

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc.
        limbs: str, required
            Name of the robot limb part, i.e. "gripper" or "arm"
    return
    ----------
        Return the ID's of the given model limbs
    """
    return tuple(joint_from_name(body, name) for name in ROBOT_GROUPS[limbs])


def set_arm_config(body, limbs, config):
    """
    Set joint configuration into the simulation environment with a given robot limb

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc.
        limbs: str, required
            Name of the robot limb part, i.e. "gripper" or "arm"
        config: list/numpy (flatten), required
            Joint configuration, that should be set. Limbs have to define 
            in Simulation/Robots/robot_utils.py in the dictionary ROBOT_GROUPS. 
    """
    [set_joint_position(body, joint, conf) for joint, conf in zip(get_joint_from_model(body, limbs), config)]


def open_gripper(body):
    """
    Open gripper in the simulation environment. Define ROBOT_GROUPS_ID first, if you are using a different Robot.

    Parameters
    ----------
       body: int, required
           body unique id, as returned by loadURDF etc.
    """
    max_limit = np.array([get_limits_of_joint_info(body, joint) for joint in ROBOT_GROUPS_ID['gripper']])[:, 1]

    for joint, conf in zip(ROBOT_GROUPS_ID['gripper'], max_limit):
        set_joint_position(body, joint, conf)


def close_gripper(body):
    """
    Close gripper in the simulation environment. Define ROBOT_GROUPS_ID first, if you are using a different Robot.

    Parameters
    ----------
       body: int, required
           body unique id, as returned by loadURDF etc.
    """
    min_limit = np.array([get_limits_of_joint_info(body, joint) for joint in ROBOT_GROUPS_ID['gripper']])[:, 0]

    for joint, conf in zip(ROBOT_GROUPS_ID['gripper'], min_limit):
        set_joint_position(body, joint, conf)


def transform_reference_systems(base_position, tcp_position, to_world=True):
    """
    Transform the reference system form the global world coordinate system to local robot coordinate 
    system or from the local robot coordinate system into the global world coordinate system.

    Parameters
    ----------
        base_position: numpy (flatten), required
            Base position in the global pybullet simulation coordinate system
        tcp_position: numpy (flatten), required
            Current [X, Y, Z] local or global coordinte position
        to_world: bool, optional
            If true, transform position in global coordinate system, else into local robot coordinate system
    return
    ----------
        Return the transformed coordinate system as a flatten numpy array
    """
    if to_world:
        position = tcp_position + base_position
    else:
        # to local reference system
        position = tcp_position - base_position

    return position


def calculate_distance(vector1, vector2):
    """
    Measures the distances between two vectors.

    Parameters
    ----------
        vector1: numpy (flatten), required
        vector2: numpy (flatten), required

    return
    ----------
        Return the distance as a flatten numpy array
    """
    return np.sqrt(np.sum((np.array(vector1) - np.array(vector2)) ** 2))


class CoordinateSystem(object):
    """
    Class for constructing and drawing coordinate systems into the pybullet simulation environment. Saves 
    all new coordinate systems into a dictionary and also can delete coordinate systems in the pybullet 
    simulation.
    """

    def __init__(self, body):
        self.body = body

        # dictionary for joint debug line
        self.joint_debug_line = {}
        # dictionary for the text of the joint debug line
        self.joint_debug_text = {}

    @staticmethod
    def get_z_point():
        return np.array([0, 0, 1])

    @staticmethod
    def get_y_point():
        return np.array([0, 1, 0])

    @staticmethod
    def get_x_point():
        return np.array([1, 0, 0])

    @staticmethod
    def transform_points(rotation_matrix, point):
        """
        Calculates the end coordinate of the direction 

        Parameters
        ----------
            rotation_matrix: numpy, required
                3x3 rotation matrix
            point: numpy, required
                Direction of the axis as a flatten numpy array
        """
        return rotation_matrix @ point

    def delete_debug_line(self, idx):
        """
        Delete coordinate system in pybullet simulation.

        Parameters
        ----------
            idx: int, required
                Index/name of the debug line that should be deleted
        """
        lines = self.joint_debug_line[idx]
        text = self.joint_debug_text[idx]

        for line, t in zip(lines, text):
            remove_debug_item(line)
            remove_debug_item(t)

    def clear_all(self):
        """
        Delete all coordinate systems in pybullet simulation and clears the dictionary.
        """
        for key in self.joint_debug_line.copy():
            self.delete_debug_line(key)

            self.joint_debug_line.pop(key)
            self.joint_debug_text.pop(key)

    def draw_coordinate_system(self, position, rotation_matrix, name, length=0.25):
        """
        Draw coordinate system into pybullet simulation and store pybullet ids in dictionaries.

        Parameters
        ----------
            position: numpy (flatten), required
                Global XYZ of the start position for the coordinate system
            rotation_matrix: numpy, required
                3x3 rotation matrix for the coordinate system orientation
            name: str, required
                Name of the coordinate system
            length: float, optional
                Length of the line that should be drawn into the simulation
        """

        x = self.transform_points(rotation_matrix, self.get_x_point()) * length
        y = self.transform_points(rotation_matrix, self.get_y_point()) * length
        z = self.transform_points(rotation_matrix, self.get_z_point()) * length

        x_line = draw_line([position, position + x], (1, 0, 0), 1)
        x_text = draw_text("x_" + str(name), position + x, (1, 0, 0))

        y_line = draw_line([position, position + y], (0, 1, 0), 1)
        y_text = draw_text("y_" + str(name), position + y, (0, 1, 0))

        z_line = draw_line([position, position + z], (0, 0, 1), 1)
        z_text = draw_text("z_" + str(name), position + z, (0, 0, 1))

        self.joint_debug_line[name] = [x_line, y_line, z_line]
        self.joint_debug_text[name] = [x_text, y_text, z_text]

    def to_rotation_matrix(self, joint):
        """
        Calculate the rotation matrix from a current joint configuration

        Parameters
        ----------
            joint: int, required
                ID number of the joint

        return
        ----------
            Return rotation matrix as a 3x3 numpy matrix
        """
        return np.array(list(matrix_from_quaternion(list(get_fkine_orientation(self.body, joint))))).reshape((3, 3))

    @staticmethod
    def convert_orientation_2_euler(rotation_matrix):
        """
        Convert rotation matrix to euler angles

        Parameters
        ----------
            rotation_matrix: numpy, required

        return
        ----------
            Return euler angles as a list
        """
        return list(euler_from_quaternion(quaternion_from_matrix(rotation_matrix)))

    @staticmethod
    def convert_orientation_2_matrix(euler):
        """
        Convert euler angles into rotation matrix

        Parameters
        ----------
            euler: list/numpy, required
                Euler angles, flatten numpy array or 3x1 numpy array

        return
        ----------
            Return 3x3 rotation matrix as a numpy array
        """
        return np.array(matrix_from_quaternion(quaternion_from_euler(euler))).reshape((3, 3))


class CoordinateSystemControl(CoordinateSystem, MultiThreading):
    def __init__(self, robot, visualization=False):
        """
        Control TCP coordinate system with app and update joint coordinate systems

        Parameters
        ----------
            robot: RobotSetup, required
                Object description of the robot model with all important information
            visualization: bool, optional
                Boolean for updating/setting joint coordinate system into the pybullet simulation
        """
        MultiThreading.__init__(self)
        CoordinateSystem.__init__(self, robot.body)

        self.body = robot.body
        self.visualization = visualization
        self.joints = all_joints(self.body)[:robot.dof]

        self.last_joints_position = get_all_joint_position(self.body)[:robot.dof]

        if self.visualization:
            # Visualize coordinate systems for all joints
            if self.num_workers >= 2:
                # divide tasks into two separate threads and draw coordinate systems
                odd_joint_thread, even_joint_thread = self.divide_coordinate_tasks(self.joints)
                self.configure_threads([self.multi_draw_coordinate_systems] * 2,
                                       [even_joint_thread, odd_joint_thread])
            else:
                for joint in self.joints:
                    joint_position = list(get_fkine_position(self.body, joint))
                    joint_orientation = self.to_rotation_matrix(joint)

                    self.draw_coordinate_system(joint_position, joint_orientation, joint + 1)

    def update_coordinate_system(self, joint_position, rotation_matrix, joint):
        """
        Delete last coordinate system of this current joint and draw new coordinate system

        Parameters
        ----------
            joint_position: list/numpy, required
                Global XYZ of the start position for the coordinate system
            rotation_matrix: numpy, required
                3x3 rotation matrix for the coordinate system orientation
            joint: int, required
                ID of the current joint
        """
        self.delete_debug_line(joint)
        self.draw_coordinate_system(joint_position, rotation_matrix, joint)

    def multi_draw_coordinate_systems(self, values):
        """
        Draw list of coordinate systems into pybullet simulation

        Parameters
        ----------
            values: list, required
                list of tuples of joint position, rotation matrix and the joint ids
        """
        for value in values:
            joint_position, rotation_matrix, joint = value[0], value[1], value[2]
            self.draw_coordinate_system(joint_position, rotation_matrix, joint)

    def multi_update_coordinate_systems(self, values):
        """
        Update list of coordinate system. Delete the current joint coordinate system and draw new one.

        Parameters
        ----------
            values: list, required
                List of tuples of joint position, rotation matrix and the joint ids
        """
        for value in values:
            joint_position, rotation_matrix, joint = value[0], value[1], value[2]
            self.update_coordinate_system(joint_position, rotation_matrix, joint)

    def divide_coordinate_tasks(self, joints_to_update):
        """
        Separate joints that should be updated

        Parameters
        ----------
            joints_to_update: list, required
                List of joint ids, that should be updated
        return
        ----------
            Return separated list of updating coordinate system tasks
        """
        odd_joint_thread = []
        even_joint_thread = []

        for joint in joints_to_update:
            joint_position = list(get_fkine_position(self.body, joint))
            joint_orientation = self.to_rotation_matrix(joint)

            if joint % 2 == 0:
                even_joint_thread.append([joint_position, joint_orientation, joint + 1])
            else:
                odd_joint_thread.append([joint_position, joint_orientation, joint + 1])
        return odd_joint_thread, even_joint_thread

    def update(self, new_joint_position):
        """
        Separate joints that should be updated in the pybullet simulation

        Parameters
        ----------
            new_joint_position: list, required
                List of all new joint positions  as flatten numpy array
        """
        if self.visualization:
            update_joints = []
            # search all updated joint positions
            for idx, (n_joint, o_joint) in enumerate(zip(new_joint_position, self.last_joints_position)):
                if n_joint != o_joint:
                    update_joints = self.joints[idx:]
                    break
            
            # check if list is not empty, before seperating task
            if update_joints:
                # we are only using two workers for updating coordinates systems
                if self.num_workers >= 2:
                    even_joint_thread, odd_joint_thread = self.divide_coordinate_tasks(update_joints)
                    self.configure_threads([self.multi_update_coordinate_systems] * 2,
                                           [even_joint_thread, odd_joint_thread])

                else:
                    for joint in update_joints:
                        joint_position = list(get_fkine_position(self.body, joint))
                        joint_orientation = self.to_rotation_matrix(joint)

                        self.update_coordinate_system(joint_position, joint_orientation, joint)

                self.last_joints_position = new_joint_position


class RobotTCPControl(CoordinateSystem):
    last_position = None
    last_orientation = None
    planning = True

    tcp_position = None
    tcp_orientation = None

    def __init__(self, robot, workspace):
        """
        Control TCP position in pybullet simulation with app

        Parameters
        ----------
            robot: RobotSetup, required
                Object description of the robot model with all important information
            workspace:  list, required
                List of the workspace parameters of the robot, that should be controlled
        """
        super().__init__(robot.body)
        self.body = robot.body

        start_position = []
        for w in workspace:
            start_position.append(w[3])

        self.update(start_position)

    def update(self, new_position):
        """
        Update TCP position in pybullet simulation

        Parameters
        ----------
            new_position: list, required
                Position and orientation as euler angle in one list
        """
        position = np.array(new_position[:3])
        orientation = self.convert_orientation_2_matrix(new_position[3:])

        if not (isinstance(self.last_position, np.ndarray) and isinstance(self.last_orientation, np.ndarray)):
            self.draw_coordinate_system(position, orientation, "target", 0.05)
            self.last_position, self.last_orientation = position.copy(), orientation.copy()
        else:
            if not (np.array_equal(self.last_position, position) and
                    np.array_equal(self.last_orientation, orientation)):
                self.delete_debug_line("target")
                self.draw_coordinate_system(position, orientation, "target", 0.05)
                self.last_position, self.last_orientation = position.copy(), orientation.copy()

    def add_new_tcp_target(self, parameters, adding_tcp=True):
        """
        Add new tcp coordinate system in pybullet simulation

        Parameters
        ----------
            parameters: list, required
                Position and orientation as euler angle in one list
            adding_tcp: bool, required
                if true, adding new tcp position for a trajectory that should be followed
        """
        position = np.array(parameters[:3]).reshape((1, 3))
        orientation = np.array(parameters[3:]).reshape((1, 3))

        if adding_tcp:
            # Save all target points for a trajectory, needed for exercise 4
            if isinstance(self.tcp_position, np.ndarray):
                self.tcp_position = np.vstack((self.tcp_position, position))
                self.tcp_orientation = np.vstack((self.tcp_orientation, orientation))
            else:
                self.tcp_position, self.tcp_orientation = position, orientation

        # draw new coordinate systems
        length = len(self.joint_debug_line)
        self.draw_coordinate_system(position.flatten(), self.convert_orientation_2_matrix(orientation.flatten()),
                                    "t_" + str(length), 0.05)

    def swap_tcp_target(self, tcp_targets):
        """
        Swap tcp targets and order the new trajectory in pybullet simulation.

        Parameters
        ----------
            tcp_targets: list, required
                List of the index of the tcp targets that should be swapped

        In pybullet it is better to delete all coordinate systems and draw new ones. Errors can be easy avoided
        """
        # delete tcp position
        self.delete_tcp_targets(False)

        # swap rows
        self.tcp_position[[tcp_targets[0], tcp_targets[1]]] = self.tcp_position[[tcp_targets[1], tcp_targets[0]]]
        self.tcp_orientation[[tcp_targets[0], tcp_targets[1]]] = self.tcp_orientation[[tcp_targets[1], tcp_targets[0]]]

        # draw tcp position in new order
        for idx in range(len(self.tcp_position)):
            self.add_new_tcp_target(np.concatenate((self.tcp_position[idx], self.tcp_orientation[idx])), False)

    def delete_single_tcp_target(self, delete_targets):
        """
        Delete tcp target in a trajectory in pybullet simulation and draw new TCP sequence

        Parameters
        ----------
            delete_targets: list, required
                List of indexes that should be deleted in the trajectory
        """
        self.delete_tcp_targets(False)

        # delete rows
        for idx in reversed(delete_targets):
            self.tcp_position = np.delete(self.tcp_position, idx, 0)
            self.tcp_orientation = np.delete(self.tcp_orientation, idx, 0)

        # update tcp target orders
        for idx in range(len(self.tcp_position)):
            self.add_new_tcp_target(np.concatenate((self.tcp_position[idx], self.tcp_orientation[idx])), False)

    def delete_tcp_targets(self, reset_tcp=True):
        """
        Delete all tcp targets or/and reset trajectory target points

        Parameters
        ----------
            reset_tcp: bool, required
                Boolean for resetting a trajectory
        """
        for key in self.joint_debug_line.copy():
            if not key == 'target':
                self.delete_debug_line(key)

                self.joint_debug_line.pop(key)
                self.joint_debug_text.pop(key)

        if reset_tcp:
            self.tcp_position = None
            self.tcp_orientation = None

    def save_path_plan(self, path):
        """
        Save trajectory.

        Parameters
        ----------
            path: str, required
                Path were the file should be saved
        """
        np.save(path, [self.tcp_position, self.tcp_orientation])

    def load_path_plan(self, path):
        """
        Load trajectory from a file and draw coordiante systems into pybullet simulation.

        Parameters
        ----------
            path: list, required
                Path were the file is saved
        """
        plan = np.load(path)
        self.tcp_position, self.tcp_orientation = plan[0], plan[1]

        # draw trajectory in pybullet simulation
        for i in range(len(self.tcp_position)):
            self.draw_coordinate_system(self.tcp_position[i],
                                        self.convert_orientation_2_matrix(self.tcp_orientation[i]),
                                        "t_" + str(i + 1), 0.05)


class UserDebugControl(object):
    # Class for setting joints in pybullet simulation. Used for exercise 2 forward kinematic
    def __init__(self, body, robot_joints):
        self.body = body
        self.robot_joints = robot_joints
    
    def update_robot_joints(self, values):
        # method for new debug joint parameters and set joint posiiton in simulation
        for idx, joint in enumerate(self.robot_joints):
            if idx == len(self.robot_joints) - 1:
                set_joint_position(self.body, joint + 1, values[-1])
            set_joint_position(self.body, joint, values[idx])


def check_costum_limits_trajectory(values, min_values, max_values):
    """
    Check limits of a given trajectory with costum limits.

    Parameters
    ----------
        values: numpy, required
            Shape of a (n, dof)-numpy array of the trajectory
        min_values: numpy, required
            Minimum values of the costum limits
        max_values: numpy, required
            Maximum values of the costum limits

    return
    ----------
        If in limits, return True

    """
    for row in values:
        if not check_custom_limits(row, min_values, max_values):
            return False
    
    return True


def random_arm_config(robot, curr_joints, i, attempts):
    """
    Add Variance to the current joint configuration of a robot arm by using attempts
    Used for searching all joint position in a trajectory. If robot reached a joint limit 
    or is in singularity, start from new random arm position by adding different strength of variance.

    Parameters
    ----------
        robot: RobotSetup, required
            Object description of the robot model with all important information
        curr_joints: numpy, required
            Current joint position as a flatten numpy array
        i: int, optional
            Current step
        attempts: int, optional
            Maximum attempts

    return
    ----------
        Return new joint configuration as a flatten numpy array
    """
    if i == 0:
        # q0 random config small changes
        config = get_sample_arm_config(robot, curr_joints,
                                       robot.joint_limits[:robot.dof], 0)
    elif 0 < i < int(attempts / 2):
        config = get_sample_arm_config(robot, curr_joints,
                                       robot.joint_limits[:robot.dof], 1)
    else:
        config = get_sample_arm_config(robot, curr_joints,
                                       robot.joint_limits[:robot.dof], 2)

    return config.copy()


def get_sample_arm_config(robot, joint_positions, limits=None, random_state=0):
    """
    Sample current joint position with no, low and medium variance to get a new start position for searching
    an inverse kinematic

    Parameters
    ----------
        robot: RobotSetup, required
            Object description of the robot model with all important information
        joint_positions: numpy, required
            Current joint position as a flatten numpy array
        limits: numpy, optional
            Limit of the robot model which clips the new joint configuration to the limits
        random_state: int, optional
            Value for no, low and medium variance to the current joint position

    return
    ----------
        Return new joint configuration as a flatten numpy array
    """
    if limits is not None:
        lower_limits, upper_limits = limits[:, 0], limits[:, 1]
    else:
        limits = get_all_joint_limits(robot.body)[:robot.dof]
        lower_limits, upper_limits = limits[:, 0], limits[:, 1]

    if random_state == 0:
        variance = 0.0
    elif random_state == 1:
        variance = 0.17
    else:
        variance = 0.34

    random_values = np.random.random(robot.dof) * variance

    joint_positions -= random_values
    clipped_joints = np.clip(joint_positions, lower_limits, upper_limits)

    return clipped_joints
