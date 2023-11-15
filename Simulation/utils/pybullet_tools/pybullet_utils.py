from Simulation.utils.logging.runtime_handler import info_logger, error_logger, debug_load_model, debug_joint_config
from Simulation.utils.logging.collision_handler import debug_collision_links
from Simulation.utils.pybullet_tools.os_utils import get_real_path
from scipy.spatial.transform import Rotation as R
from Simulation.utils.utils import rad_2_grad
from itertools import product
import pybullet as p
import numpy as np
import time
import sys
import os


CLIENT = 0

GRAVITY = -9.81
URDF_FLAGS = [p.URDF_USE_INERTIA_FROM_FILE,
              p.URDF_USE_SELF_COLLISION,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS]

# in kg
STATIC_MASS = 1
COLLISION_DISTANCE = 0.0


def get_data_path():
    import pybullet_data
    return pybullet_data.getDataPath()


def add_data_path(data_path=None):
    if data_path is None:
        data_path = get_data_path()
    p.setAdditionalSearchPath(data_path)
    return data_path


def get_urdf_flags(cache=False):
    """
    Enable flags for loading model into pybullet simulation

    Parameters
    ----------
        cache: bool, optional
            Enable/Disable URDF graphics shapes

    return
    ----------
        Simulation flags (see comments which already initialized)

    """
    # by default, Bullet disables self-collision
    # URDF_USE_IMPLICIT_CYLINDER
    # URDF_INITIALIZE_SAT_FEATURES
    # URDF_ENABLE_CACHED_GRAPHICS_SHAPES seems to help
    # but URDF_INITIALIZE_SAT_FEATURES does not (might need to be provided a mesh)
    # flags = p.URDF_INITIALIZE_SAT_FEATURES | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    flags = 0
    if cache:
        flags |= p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES  # merge with or command
    return flags


def set_client(client):
    """
    Set pybullet client id

    Parameters
    ----------
        client: int, required
            pybullet simulation client
    """
    global CLIENT
    CLIENT = client


def get_client():
    """
    Returns pybullet simulation client

    return
    ----------
        Client id (int)

    """
    global CLIENT
    return CLIENT


###############################################################################
###############################################################################
def is_connected():
    """
    Check if simulation is still active

    return
    ----------
        Boolean connection. If connection is still active
    """
    return p.getConnectionInfo(physicsClientId=CLIENT)['isConnected']


def get_connection(client=None):
    """
    Get connection of a specific simulation. Check connection of the client

    Parameters
    ----------
        client: int, required
            pybullet simulation client
    return
    ----------
        Boolean connection. If connection is still active
    """
    if client is not None:
        return p.getConnectionInfo(physicsClientId=client)['connectionMethod']
    else:
        return p.getConnectionInfo(physicsClientId=get_client())['connectionMethod']


def has_active_gui():
    """
    Check if pybullet simulation is active. 

    return
    ----------
        If active pybullet simulation gui, return True

    """
    return get_connection(get_client()) == p.GUI


def step_simulation(sleep_time=0.0):
    """
    Set step in simulation. Updates simulation every step

    Parameters
    ----------
        sleep_time: float, optional
            Sleep step in simulation
    """
    p.stepSimulation(physicsClientId=CLIENT)
    time.sleep(sleep_time)


def realtime_simulation(enable=False):
    """
    Set pybullet simulation to realtime. Calling stepSimulation is needed now for MotionControl.

    Parameters
    ----------
        enable: bool, optional
            Activates realtime simulation in pybullet
    """
    p.setRealTimeSimulation(
        enableRealTimeSimulation=enable, physicsClientId=CLIENT)


def activate_gravity(gravity=None):
    """
    Activates gravity in the simulation

    Parameters
    ----------
        gravity: float, optinal
            Set individual gravity to the simulation. If not set, use standard earth gravity
    """
    if gravity is not None:
        p.setGravity(0, 0, gravity)
    else:
        p.setGravity(0, 0, GRAVITY)


def load_pybullet_model(filename, fixed_base=False, scale=1.0):
    """
    Load model into simulation

    Parameters
    ----------
        filename: str, required
            Relative path of project head

        fixed_base: bool, optional
            Set model or model base as fixed (cannot be moved)

        scale: float, optional
            Set scale of the object in the simulation

    return
    ----------
        Return an integer value for calling the object in the simulation
    """
    with LockRenderer():
        filename = get_real_path(filename)

        if filename.endswith('.urdf') or filename.endswith('.xacro'):
            try:
                flags = get_urdf_flags()
                model = p.loadURDF(filename,
                                   useFixedBase=fixed_base,
                                   flags=flags, globalScaling=scale,
                                   physicsClientId=CLIENT)

                info_logger("Path of the model found: " + str(filename), 1)
                return model
            except ValueError:
                error_logger("File not found", 1)
                debug_load_model(
                    load_pybullet_model.__name__, __file__, filename)
                raise ValueError("Error in Filename", filename)

        else:
            error_logger("Given file does not end with '.urdf'", 1)
            debug_load_model(load_pybullet_model.__name__, __file__, filename)

            raise ValueError("Error in filename-ending: ", filename)


def quaternion_from_euler(orientation):
    """
    Transform euler angles to quaternion angles

    Parameters
    ----------
        orientation: tuple/numpy/list, required
            Euler angles as a [3x1] or a flatten numpy array 

    return
    ----------
        Returns a [4x1] tuple of quaternion angles
    """
    return p.getQuaternionFromEuler(orientation)


def euler_from_quaternion(orientation):
    """
    Transform quaternion angles to euler angles

    Parameters
    ----------
        orientation: tuple/numpy/list, required
            Quaternion angles as a [4x1] list or a flatten numpy array

    return
    ----------
        Returns a [3x1] tuple of euler angles
    """
    return p.getEulerFromQuaternion(orientation)


def matrix_from_quaternion(quaternion):
    """
    Transform quaternion angles to rotation matrix

    Parameters
    ----------
        quaternion: list [4x1], required

    return
    ----------
        orientation: list [3x3]
    """
    return p.getMatrixFromQuaternion(quaternion)


def quaternion_from_matrix(matrix):
    """
    Transform quaternion angles to rotation matrix

    Parameters
    ----------
        matrix: numpy [3x3], required

    return
    ----------
        orientation: numpy [3x1]
    """
    return R.from_matrix(matrix).as_quat()


def connect(c=p.GUI, dt=1./60.):
    """
    Render pybullet simulation. p.GUI activates simulation rendering. p.DIRECT activates simulation, without rendering

    The simulation will automatically set to a specific time step. The pybullet simulation will automatically update.
    No step_simulation is needed in MotionControl

    Parameters
    ----------
        c: int, optional
            render option
        dt: float, optional
            time step for updating the simulation step
    """
    p.connect(c)
    debug_visualizer()

    set_time_step(dt)


def debug_visualizer():
    """
    Pybullet debug visualizer option.
    """
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


def disconnect():
    """
    Disconnect and closing pybullet simulation
    """
    p.disconnect()


# POSITION AND ORIENTATION
def set_position(body, position):
    """
    Set base position of a body into the simulation

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

        position: list [3x1], required
            [x, y, z]-position of the global pybullet simulation
    """
    (_, orientation) = p.getBasePositionAndOrientation(body)
    set_position_and_orientation(body, position, euler_from_quaternion(orientation))


def get_position(body):
    """
    Get global pybullet position of a pybullet body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Retuns the [X, Y, Z] position of the body as a list.
    """
    (position, _) = p.getBasePositionAndOrientation(body)
    return position


def set_orientation(body, orientation):
    """
    Set orientation of a body in the pybullet simulation

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

        orientation: list [3x1], required
            Euler angle in respect of the  pybullet simulation coordinate systems
    """
    orientation = quaternion_from_euler(orientation)
    position = get_position(body)

    set_position_and_orientation(body, position, orientation)


def get_orientation(body):
    """
    Get orientation of a pybullet body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Returns the orientation as a [4x1] quaternion angle as a list
    """
    (_, orientation) = p.getBasePositionAndOrientation(body)
    return orientation


def set_position_and_orientation(body, position, orientation):
    """
    Set position and orientation of a pybullet body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

        position: list [3x1], required
            [x, y, z]-Position

        orientation: list [3x1], required
            Orientation in euler angles
    """
    orientation = quaternion_from_euler(orientation)
    p.resetBasePositionAndOrientation(body, position, orientation)


def get_joint_info(body, joint):
    """
    Return information of a specific joint.

    JointIndex, JointName, JointType, uIndex, flags, jointDamping, jointFriction, jointLowerLimits, jointUpperLimits,
    jointMaxForce, jointMaxVelocity, linkName, jointAxis, parentFramePos, parentFrameOrn, parentIndex

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

        joint: int, required
            joint id of the body

    return
    ----------
        JointInfo: dict
    """
    return p.getJointInfo(body, joint, physicsClientId=CLIENT)


def get_joint_name(body, joint):
    """
    Return joint name from joint info.

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

        joint: int, required
            joint id of the body

    return
    ----------
        Return joint name name as a string
    """
    return p.getJointInfo(body, joint, physicsClientId=CLIENT)[1].decode('UTF-8')


def get_link_name(body, link):
    """
    Return link name from joint info.

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

        link: int, required
            link id of the body

    return
    ----------
        Return joint link name as a string
    """
    return p.getJointInfo(body, link, physicsClientId=CLIENT)[12].decode('UTF-8')


def get_joints(body):
    """
    Get all joints ids as a list.

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return all joint ids as a list
    """
    return list(range(get_number_of_joints(body)))


get_links = get_joints


def get_limits_of_joint_info(body, joint):
    """
    Return joint limits of a body.

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

        joint: int, required
            Individual joint of the model
    return
    ----------
        Return joint minimum and maximum limit as a list 
    """
    return [p.getJointInfo(body, joint, physicsClientId=CLIENT)[8],
            p.getJointInfo(body, joint, physicsClientId=CLIENT)[9]]


def get_all_joint_limits(body):
    """
    Return all joint limits of a body.

    The difference between get_all_joint_limits and get_all_limbs_limits is, that in
    get_all_limbs_limit we are only taking joint, that can be set. I. e. joint_hand limits is sometimes not needed
    for the description

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
    return
    ----------
        Return all joint limits as a numpy array with a shape of [dof, 2]
    """
    number_of_joints = get_number_of_joints(body)
    limits = []
    for joint in range(number_of_joints):
        limits.append(get_limits_of_joint_info(body, joint))

    return np.array(limits)


def get_joint_limits_from_name(body, joint_name_dict):
    """
    Return all joint limits of a body.

    Robot can be described as a dictionary, Arm and Gripper (see robot_utils.py for an example)

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

        joint_name_dict: dict, required
            Dictionary of the joint-names

    return
    ----------
        Return all joint limits as a list with a shape of [dof, 2]
    """
    limits = []

    # get all id-joints from the robot body
    joints = all_joints(body)
    dict_joints = {}
    for joint in joints:
        dict_joints[get_joint_name(body, joint)] = joint

    for joint_name in joint_name_dict:
        for name in joint_name_dict:
            if joint_name == name:
                joint_limit = get_limits_of_joint_info(body, dict_joints[name])
                limits.append(joint_limit)

    return limits


def get_all_limbs_limits(body):
    """
    Return only limbs joint limits of a body.

    The difference between get_all_joint_limits and get_all_limbs_limits is, that in
    get_all_limbs_limit we are only taking joint, that can be set. I. e. joint_hand limits is sometimes not needed
    for the description

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return all joint limits as a numpy array with a shape of [dof, 2]
    """
    from Simulation.Robots.robot_utils import ROBOT_GROUPS_ID

    joints = []
    for name in ROBOT_GROUPS_ID:
        for element in ROBOT_GROUPS_ID[name]:
            joints.append(element)

    joint_limits = []
    for joint in joints:
        joint_limits.append(get_limits_of_joint_info(body, joint))

    # Attention: if finger1 is changed, then finger2 gets the same status. in limbs finger2 it is not included
    return np.asarray(joint_limits[:-1])


def get_eef_index(body):
    """
    Return eef joint index. Need right urdf description

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        eef index: int
        """
    return get_number_of_joints(body) - 1


def get_number_of_joints(body):
    """
    Return number of joints of a body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return number of joints as an integer
    """
    return p.getNumJoints(body, physicsClientId=CLIENT)


get_number_of_links = get_number_of_joints


def all_joints(body):
    """
    Return all joints as a list

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
    """
    return list(range(get_number_of_joints(body)))[:-1]


all_links = all_joints


def get_arm_gripper_joints(body):
    """
    Return joint ids of a pybullet body form a specified dictionary ROBOT_GROUPS defined in robot_primitives.py.

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
    return
    ----------
        Return gripper joints as a list
    """
    from Simulation.Robots.robot_primitives import ROBOT_GROUPS
    joint_ids = []
    for group in ROBOT_GROUPS:
        joint_names = ROBOT_GROUPS[group]

        for name in joint_names:
            joint_ids.append(joint_from_name(body, name))

    return joint_ids


def get_joint_config(body, joint):
    """
    Return all information from a joint (joint-limit, current joint-velocity, ect). See pybullet documentation 

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joint: int, required
            joint id of the body
    return
    ----------
        Return the joint-state of an individual joint as a list
    """
    return p.getJointState(body, joint, physicsClientId=CLIENT)


def get_all_joint_config(body):
    """
    Return all information from all joints (joint-limit, current joint-velocity, ect). See pybullet documentaiton

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return the joint-state of an individual joint as a dictionary
    """
    joints = all_joints(body)
    config = p.getJointStates(body, joints)

    return config


def get_joint_velocity(body, joint):
    """
    Return current joint velocity

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joint: int, requrired
            joint id of the body

    return
    ----------
        Return velocity of a joint as a float

    """
    return p.getJointState(body, joint, physicsClientId=CLIENT)[1]


def get_all_joint_velocity(body):
    """
    Return all velocities of all joints of the body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return velocities as a list

    """
    joints = get_joints(body)

    velocity = []
    for joint in joints:
        velocity.append(get_joint_velocity(body, joint))

    return velocity


def get_all_joint_forces(body):
    """
    Return all forces of all joints of the body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return forces as a list

    """
    joint_states = get_all_joint_config(body)
    joint_infos = [get_joint_info(body, joint) for joint in all_joints(body)]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]

    torques = [state[3] for state in joint_states]

    return torques


def get_joint_position(body, joint):
    """
    Return the current joint position/angle of the given joint

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joint: int, requrired
            joint id of the body

    return
    ----------
        Return the current joint position as a float

    """
    return p.getJointState(body, joint, physicsClientId=CLIENT)[0]


def get_all_joint_position(body):
    """
    Return the joint position/angle of all joints in the body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return the current joint position as a list

    """
    position = []

    joints = all_joints(body)

    for joint in joints:
        position.append(get_joint_position(body, joint))

    return position


def get_limb_joints():
    """
    Return all joint and gripper ids of a predefined dictionary ROBOT_GROUPS_ID in Simulaiton/Robots/robot_utils
    All fingers in the Gripper are defined as one id.

    return
    ----------
        Return all joint position as a list

    """
    from Simulation.Robots.robot_utils import ROBOT_GROUPS_ID
    joints = []
    for group in ROBOT_GROUPS_ID:
        for joint in ROBOT_GROUPS_ID[group]:
            joints.append(joint)

    # Attention: if finger1 is changed, then finger2 gets the same status. in limbs finger2 it is not included
    return joints[:-1]


def get_limb_positions(body):
    """
    Return all joint and gripper position of a predefined dictionary ROBOT_GROUPS_ID in Simulaiton/Robots/robot_utils
    All fingers in the Gripper are defined as one id.

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return all joint position as a list
    """
    joints = get_limb_joints()

    positions = []
    for joint in joints:
        positions.append(get_joint_position(body, joint))

    # Attention: if finger1 is changed, then finger2 gets the same status. in limbs finger2 it is not included
    return positions


# Check if there is a joint with this specific name in the robot
def joint_from_name(body, name, save_values=False):
    """
    Return ID of a joint-name

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        name: str, required
            Name of the joint defined in the URDF file
        save_values: bool, optional
            Boolean for saving all joint ids in a file

    return
    ----------
        Return the ID of the joint-name as an int
    """
    # get all id-joints from the robot body
    joints = all_joints(body)

    dict_joints = {}
    for joint in joints:
        dict_joints[get_joint_name(body, joint)] = joint

    if save_values:
        # in get_joint_name comment out #[0].decode('UTF-8')
        save_all_joint_names_and_links(dict_joints)

    # we have to check, if name is in joint space
    for joint in dict_joints:
        if joint == name:
            return dict_joints[name]

    error_logger("Name of the joint not found", 4)
    raise ValueError("Name of the joint not found", body, name)


def link_from_name(body, name, save_values=False):
    """
    Return ID of a joint-name

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        name: str, required
            Individual pybullet name of the model
        save_values: bool, optional
            Boolean for saving all joint ids in a file
    
    return
    ----------
        Return the ID of the joint-name as an int
    """
    # get all id-joints from the robot body
    joints = all_links(body)

    dict_joints = {}
    for joint in joints:
        dict_joints[get_link_name(body, joint)] = joint

    if save_values:
        # in get_joint_name comment out #[0].decode('UTF-8')
        save_all_joint_names_and_links(dict_joints)

    # we have to check, if name is in joint space
    for joint in dict_joints:
        if joint == name:
            return dict_joints[name]

    error_logger("Name of the joint not found", 4)
    raise ValueError("Name of the joint not found", body, name)


def set_joint_positions(body, joints, config):
    """
    Set multiple joint positions/angles in pybullet simulation

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joints: list, required
            Individual model joint IDs
        config: list, required
            Values/Position of the joint ID in radians
    """
    for joint, conf in zip(joints, config):
        set_joint_position(body, joint, conf)


def set_joint_position(body, joint, config):
    """
    Set single joint of the body in the pybullet simulation

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joint: int, required
            joint id of the body 
        config: float, required
            Value/Position of the joint ID in radians

    """
    p.resetJointState(body, joint, config)
    check_joint_limits(body, joint)


###############################################################################
###############################################################################
def check_joint_limits(body, joint):
    """
    Load limits of a given joint from URDF and check if joint is in limits

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joint: int, required
            joint id of the body

    return
    ----------
        Return True if its in joint-limits, else False

    """
    # This method is when joint is already set
    # this function should raise an error and write it into log file
    # in limits only debug it
    # joint is in limit --> warning
    # joint is outside of limit --> error

    limits = get_limits_of_joint_info(body, joint)
    joint_pos = get_joint_config(body, joint)[0]

    if limits[0] < joint_pos < limits[1]:
        # info_logger("Current Joint is in Limits", 4)
        # debug_joint_config(model,
        #                    "panda",
        #                    joint,
        #                    get_joint_name(body, joint),
        #                    limits,
        #                    joint_pos
        # )
        return True
    elif joint_pos == limits[0] or joint_pos == limits[1]:
        # warning_logger("Current Joint is on max/min limit", 4)
        # debug_joint_config(body,
        #                    "panda",
        #                    joint,
        #                    get_joint_name(body, joint),
        #                    limits, joint_pos
        # )
        return True
    else:
        error_logger('Robot destroyed, Invalid Joint limit', 4)
        debug_joint_config(body,
                           "panda",
                           joint,
                           get_joint_name(body, joint),
                           limits,
                           joint_pos
                           )
        return False
        # raise ValueError('Congratulation: You destroyed the robot')


def is_in_joint_limits(body, joints):
    """
    Check if current robot configuration of all the joints is in limits

    Parameters
    ----------
    body: int, required
        body unique id, as returned by loadURDF etc
    joints: list, required
        List of all the joint IDs, that should be checked

    return
    ----------
        Return True if its in joint-limits, else False

    """
    for joint in joints:
        in_limits = check_joint_limits(body, joint)

        if not in_limits:
            return False
    return True


def check_limits(joints, values, limits):
    """
    Check if values is in costum limits and raise error otherwise

    Parameters
    ----------
        joints: list, required
            List of all current joints angle values that should be checked
        values: list, required
            List of all joint values, that should be checked
        limits: list, required
            One dimensional limit list
    """
    for joint, value, limit in zip(joints, values, limits):
        if -limit > value > limit:
            raise ValueError(
                "Velocity Joint {} Violation; Limits: [{}, {}]; Current Velocity: {}".format(
                    joint, -limit, limit, value))


def check_velocity(body, joints):
    """
    Check if current joint velocity and raise error otherwise

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joints: list, required
            List of all the joint IDs, that should be checked
    """
    _, curr_vel, _ = get_current_motor_joint_state(body)
    limits = get_all_max_velocities(body, joints)

    for joint, c_vel, limit in zip(joints, curr_vel, limits):
        if -limit > c_vel > limit:
            raise ValueError(
                "Velocity Joint {} Violation; Limits: [{}, {}]; Current Velocity: {}".format(
                    joint, -limit, limit, c_vel))


def check_forces(body, joints):
    """
    Check if current joint forces and raise error otherwise

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joints: list, required
            List of all the joint IDs, that should be checked
    """
    _, _, curr_forces = get_current_motor_joint_state(body)
    limits = get_all_max_forces(body, joints)

    for joint, c_force, limit in zip(joints, curr_forces, limits):
        if -limit > c_force > limit:
            raise ValueError(
                "Velocity Joint {} Violation; Limits: [{}, {}]; Current Velocity: {}".format(
                    joint, -limit, limit, c_force))


def is_in_never_collision(body, link1, link2, **kwargs):
    """
    Check if a collision appears, but can be negligible. In Pybullet also appears when two links are connected.
    These can be ignored. See Simulation.Robots.never_collision

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        link1: int, required
            Link ID of the model
        link2: int, required
            Link ID of the model
        **kwargs: 
            list of never kollision

    return
    ----------
        If link1 and link 2 is in never_collision, return true, else false
    """
    for key, value in kwargs.items():
        for links in value[key]:
            l1, l2 = link_from_name(body, links[0]), link_from_name(body, links[1])

            # we have to check both version, because of itertools.product
            if (link1, link2) == (l1, l2) or (link1, link2) == (l2, l1):
                return True

    return False


def pairwise_link_collision(body1, link1, body2, link2=None, **kwargs):
    """
    Check if a collision appears of two objects in the simulation

    Parameters
    ----------
        body1: int, required
            body unique id, as returned by loadURDF etc
        link1: int, required
            Link ID of the model
        body2: int, required
            body unique id, as returned by loadURDF etc
        link2: int, required
            Link ID of the model
        **kwargs: 
            list of never kollision

    return
    ----------
        Return True, when collision
    """
    # **kwargs for never collisions
    if link2 is None:
        link2 = all_links(body2)

    pair_collisions = []
    for link_combination in product(link1, link2):
        l1, l2 = link_combination[0], link_combination[1]
        if (l1 == l2) and (body1 == body2):
            continue
        elif is_in_never_collision(body1, l1, l2, kwargs=kwargs) and (
                body1 == body2):
            continue
        else:
            is_collision = link_collision(body1, l1, body2, l2)
            if is_collision:
                link_names = (
                    get_link_name(body1, l1), get_link_name(body2, l2))
                if ((link_names not in pair_collisions) and (
                        link_names[::-1] not in pair_collisions)):
                    pair_collisions.append(link_names)

    if len(pair_collisions) != 0:
        error_logger("Links collided")
        debug_collision_links(body1, body2, pair_collisions)

        return True
        # raise ValueError("Links collided --> See in log-file")
    else:
        return False


def link_collision(body1, link1, body2, link2=None, distance=COLLISION_DISTANCE):
    """
    Check link collision of two objects in the pybullet simulation with individual links 

    Parameters
    ----------
        body1: int, required
            body unique id, as returned by loadURDF etc
        link1: int, required
            Link ID of the model
        body2: int, required
            body unique id, as returned by loadURDF etc
        link2: int, required
            Link ID of the model
        distance: float, optional
            Maximum distance of a link collision

    return
    ----------
        If collision, return true
    """
    if link2 is None:
        return len(p.getClosestPoints(bodyA=body1,
                                      bodyB=body2,
                                      distance=distance,
                                      linkIndexA=link1,
                                      physicsClientId=CLIENT)) != 0
    else:
        return len(p.getClosestPoints(bodyA=body1,
                                      bodyB=body2,
                                      distance=distance,
                                      linkIndexA=link1,
                                      linkIndexB=link2,
                                      physicsClientId=CLIENT)) != 0


def object_collision(body1, body2, visualization=False, distance=COLLISION_DISTANCE):
    """
    Check collision of two objects in the pybullet simulation and visualize it in the simulation

    Parameters
    ----------
        body1: int, required
            body unique id, as returned by loadURDF etc
        body2: int, required
            body unique id, as returned by loadURDF etc
        visualization: bool, optional
            Visualize collision and draw collision line in pybullet simulation
        distance: float, optional
            Maximum distance of a link collision

    return
    ----------
        If collision, return true

    """
    collision_results = list(
        p.getClosestPoints(bodyA=body1,
                           bodyB=body2,
                           distance=distance,
                           physicsClientId=CLIENT))

    if len(collision_results) == 0:
        return False  # no_collision
    else:
        for idx, value in enumerate(
                collision_results):  # delete base connection to another object
            if value[3] == -1:
                collision_results.pop(idx)

        if visualization:
            for collision in collision_results[1:]:
                body, link = collision[1], collision[3]
                draw_collision_line(collision_line_size(collision[5], size=0.8),
                                    color=(1, 0.5, 0.5), width=2)
                draw_text_to_collision_line(
                    "Collision in Body [{}] with link {}".format(body, link),
                    collision_line_size(collision[5], size=0.9), color=(0, 0, 1))

        if len(collision_results) != 0:
            print("Collision of bodies: {}, {}".format(body1, body2))  # TODO debug it
            return True
        else:
            return False


###############################################################################

def collision_line_size(collision, size=0.01):
    """
    Compute end position of the collision line in the pybullet simulation.

    Parameters
    ----------
        collision: numpy, required
            start value of the [X, Y, Z] Position of the collision
        size: float, optional
            line size of the collision
    
    return
    ----------
        return start- and end-position of the collision line

    """
    axis = np.zeros(len(collision))
    axis[2] = 1.0  # in z-axis
    end_pos = np.array(collision) + size / 1 * axis

    return [collision, end_pos]


def draw_collision_line(positions, color, width, lifetime=0):
    """
    Draw collision line in the pybullet simulation

    Parameters
    ----------
        positions: bool, required
            start- and end-position of the collision line. Position values are in [X, Y, Z]-
            coordinate systems as flatten numpy arrays
        color: tuple, required
            RGB color value
        width: float, required
            Line width of the line
        lifetime: int, optional 
            Duration of collision line, set in simulation
    """
    p.addUserDebugLine(positions[0], positions[1], color, lineWidth=width,
                       lifeTime=lifetime)


def draw_text_to_collision_line(collision_text, text_position, color,
                                text_size=0.7, lifetime=0):
    """
    Write text to a collision line

    Parameters
    ----------
        collision_text: str, required
            Text of the collision
        text_position: list, required
            End-position of the collision text
        color: list, required
            RGB color value
        text_size: float, optional
            Text size of the collision
        lifetime: int, optional
            Duration of collision text, set in simulation

    """
    p.addUserDebugText(text=collision_text, textPosition=text_position[1],
                       textColorRGB=color,
                       textSize=text_size, lifeTime=lifetime,
                       physicsClientId=CLIENT)


def draw_line(positions, color=(1, 0, 0), width=1.5, lifetime=0):
    """
    Draw line in Simulation and return individual user debug line id of in the simulation.
    With this id, the line can be deleted

    Parameters
    ----------
        positions: list, required
            start- and end-position of the collision line. Position values are in [X, Y, Z]-coordinate
            systems as flatten numpy arrays
        color: list, required
            RGB value color
        width: float, optional
            Width of the line
        lifetime: int, optional
            Duration of collision text, set in simulation
    return
    ----------
        Return user debug ID of the line

    """
    return p.addUserDebugLine(positions[0], positions[1], color, lineWidth=width, lifeTime=lifetime)


def draw_text(text, text_position, color, text_size=1., lifetime=0):
    """
    Draw line in Simulation and return individual user debug line id of in the simulation.
    With this id, the line can be deleted

    Parameters
    ----------
        text: str, required
            Text, that should be draw in the simulation
        text_position: list, required
            XYZ-position 
        color: list, required
            RGB value color
        text_size: float, optional
            Width of the line
        lifetime: int, optional
            Duration of collision text, set in simulation

    return
    ----------
        Return user debug ID of the text

    """
    return p.addUserDebugText(text=text, textPosition=text_position, textColorRGB=color,
                              textSize=text_size, lifeTime=lifetime, physicsClientId=CLIENT)


def remove_all_debug_items():
    """
    Remove and delete all debug items in simulation
    """
    p.removeAllUserDebugItems(physicsClientId=CLIENT)


def remove_debug_item(debug_item):
    """
    Remove/delete user debug item in simulation

    Parameters
    ----------
        debug_item: int, required
            Simulation user debug ID of the text or line

    """
    p.removeUserDebugItem(debug_item, physicsClientId=CLIENT)


###############################################################################

def load_model(filename, position=None, orientation=None, pose=None,
               fixed_base=False, **kwargs):
    """
    Load body in simulation with a specific position and orientation

    Parameters
    ----------
        filename: str, required
            Path of the body that should be loaded into the simulation
        position: list/numpy, optinal
            3x1 vector for [x, y, z]-position
        orientation: list/numpy, optional
            3x1 vector for [x, y, z]-orientation as euler angles
        pose: tuple, optional
            (position, orientation)-list
        fixed_base: bool, optional
            object can be moved in simulation
        **kwargs:    

    return
    ----------
        Return individual simulation ID for the model 

    """
    filename = get_real_path(filename)
    body = load_pybullet_model(filename, fixed_base=fixed_base, **kwargs)

    if pose is not None:
        (position, orientation) = pose
        set_position_and_orientation(body, position, orientation)

    if position is not None:
        set_position(body, position)

    if orientation is not None:
        set_orientation(body, orientation)

    return body


def save_all_joint_names_and_links(dict_joints):
    """
    Save all joint and link names in a CSV file

    Parameters
    ----------
        dict_joints: dict, required
            Discription of the robot model

    """
    import csv
    path = get_real_path("/Simulation/Data/joint_dictionary.csv")
    a_file = open(path, "w")
    writer = csv.writer(a_file)

    for key, value in dict_joints.items():
        writer.writerow([key, value])

    a_file.close()


# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/14797594#14797594
# https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262

class HideOutput(object):
    """
    A context manager that block stdout for its scope, usage:
    with HideOutput():
        os.system('ls -l')
    """
    DEFAULT_ENABLE = True

    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)
        os.close(self._oldstdout_fno)  # Added


class Saver(object):
    def restore(self):
        raise NotImplementedError()

    def __enter__(self):
        # TODO: move the saving to enter?
        pass

    def __exit__(self, type, value, traceback):
        self.restore()


def set_renderer(enable):
    """
    Set pybullet renderer. Enables adding object faster

    Parameters
    ----------
        enable: int, required

    """
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable), physicsClientId=CLIENT)


class LockRenderer(Saver):
    # disabling rendering temporary makes adding objects faster

    def __init__(self, lock=True):
        self.renderer = has_active_gui()
        if lock:
            set_renderer(enable=False)

    def restore(self):
        set_renderer(enable=self.renderer)


# for grasps
def motor_control_individual(body, joints, q_pos_desired, control_mode,
                             position_gain, velocity_gain, dt=1. / 240.):
    """
    Set stepwise motor control of a list of joints. Used for grasping an object in pybullet simulation. 

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joints: list, required
            List of joints ids
        q_pos_desired: list/numpy, required 
            target joint position
        control_mode: int, required
            Type of control mode: Position Control, Velcoity Control, Torque Control
        position_gain: list, required
            Gain of the position
        velocity_gain: list, required
            Gain of the velocity
        dt: float, optional
            timestep of the simulation that should be paused

    """
    for joint, t_pos, p_gain, v_gain in zip(joints, q_pos_desired, position_gain, velocity_gain):
        p.setJointMotorControl2(body, joint, control_mode, t_pos,
                                positionGain=p_gain, velocityGain=v_gain, physicsClientId=CLIENT)
    step_simulation(sleep_time=dt)


def motor_control(body, joints, control_mode, q=None, dq=None, forces=None, position_gain=None,
                  velocity_gain=None):
    """
    Stepwise motor control in pybullet simulation with q, qd, or forces are required. 
    Depends on control_mode. Joints and q, qd or forces has to be equal in shape

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joints: list, required
            List of all joints 
        control_mode: int, required
            Type of control mode: Position Control, Velcoity Control, Torque Control
        q: list, optional
            Target joint position as a flatten numpy array
        dq: list, optional
            Target joint velocity as a flatten numpy array
        forces: flaot, optional
            target joint force as a flatten numpy array
        position_gain: list, optional
            Gain of the position
        velocity_gain: list, optional
            Gain of the velocity    
    """
    # explicit	PD control requires	small timestep
    # timeStep = 1. / 600.

    if isinstance(forces, np.ndarray):
        p.setJointMotorControlArray(body, joints, control_mode,
                                    forces=forces[:7], physicsClientId=CLIENT)
    elif isinstance(q, np.ndarray) and isinstance(dq, np.ndarray):
        p.setJointMotorControlArray(body, joints, control_mode, q, dq,
                                    positionGains=position_gain, velocityGains=velocity_gain,
                                    physicsClientId=CLIENT)
    else:
        p.setJointMotorControlArray(body, joints, control_mode, q,
                                    positionGains=position_gain, velocityGains=velocity_gain,
                                    physicsClientId=CLIENT)


def get_current_motor_joint_state(body):
    """
    Get current motor joints states, like position, velocity and forces

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc

    return
    ----------
        Return current joint, velocity and torques of the joints as a numpy array

    """
    # Notice: if torque control, joint_torques are zero
    # from manual:  If you use TORQUE_CONTROL then theapplied 
    # joint motor torque is exactly what you provide, so there is no 
    # need to report it separately
    joint_states = get_all_joint_config(body)
    joint_infos = [get_joint_info(body, joint) for joint in all_joints(body)[:7]]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]

    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]

    return np.array(joint_positions), np.array(joint_velocities), np.array(joint_torques)


def change_dynamics_gripper(body, link_list, lateral_friction=1.0, spinning_friction=1.0, rolling_friction=0.0001,
                            friction_anchor=True):
    """
    Change dynamics of the gripper

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        link_list: list, required
            List of the links ids
        lateral_friction: float, optional
            lateral (linear) contact friction 
        spinning_friction: float, optional
            torsional friction around the contact normal
        rolling_friction: float, optional
            torsional friction orthogonal to contact normal
        friction_anchor: bool, optional
            enable or disable a friction anchor: positional friction correction (disabled by default, unless set in
            the URDF contact section)

    """
    for i in link_list:
        p.changeDynamics(body, i, lateralFriction=lateral_friction, spinningFriction=spinning_friction,
                         rollingFriction=rolling_friction, frictionAnchor=friction_anchor)


def create_constraint(body, link1=9, link2=10, joint_type=p.JOINT_GEAR,
                      joint_axis=(0, 0, 0), parent_frame_position=(0, 0, 0),
                      child_frame_position=(0, 0, -0.05)):
    """
    createConstraint allows you to connect specific links of bodies to close those loops (see pybullet documentation)

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        link1: int, optional
            parent body unique id
        link2: int, optional
            parent link index (or -1 for the base)
        joint_type: int optional
            joint type: JOINT_PRISMATIC, JOINT_FIXED, JOINT_POINT2POINT, JOINT_GEAR
        joint_axis: tuple, optional
            joint axis, in child link frame - vec3
        parent_frame_position: tuple, optional
            position of the joint frame relative to parent center of mass frame.
        child_frame_position: tuple, optional
            position of the joint frame relative to a given child center of mass frame
            (or world origin if no child specified)
    return
    ----------
        Return an unique id integer, that can be used to change or remove the constraint.
    """
    return p.createConstraint(body, link1, body, link2, jointType=joint_type, jointAxis=joint_axis,
                              parentFramePosition=parent_frame_position, childFramePosition=child_frame_position)


def change_constraint(cid, gear_ratio=-1, erp=0.1, max_forces=50):
    """
    changeConstraint allows you to change parameters of an existing constraint

    Parameters
    ----------
        cid: int, required
            unique id returned by createConstraint
        gear_ratio: float, optional
            the ratio between the rates at which the two gears rotate
        erp: float, optional
            constraint error reduction parameter
        max_forces: float, optional
            maximum force that constraint can apply
    """
    p.changeConstraint(cid, gearRatio=gear_ratio, erp=erp, maxForce=max_forces)


def remove_constraint(cid):
    """
    remove constraint allows you to remove parameters of an existing constraint

    Parameters
    ----------
        cid: int, required
            unique id returned by createConstraint.
    """
    p.removeConstraint(cid)


def get_max_velocity(body, joint):
    """
    Return the maximum velocity of a joint

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joint: int, required
            joint id of the body
    return
    ----------
        Return velocity of a joint as a float
    """
    return get_joint_info(body, joint)[11]


def get_all_max_velocities(body, joints):
    """
    Return all maximum velocities of the body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joints: list, required
            list of joint ids
    return
    ----------
        Return all maximum velocities as a list
    """
    max_velocities = []
    for joint in joints:
        vel = get_max_velocity(body, joint)
        max_velocities.append(vel)

    return max_velocities


def get_max_force(body, joint):
    """
    Return the maximum force of a joint

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joint: int, required
            joint id of the body

    return
    ----------
        Return velocity of a joint as a float

    """
    return get_joint_info(body, joint)[10]


def get_all_max_forces(body, joints):
    """
    Return all maximum forces of the body

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        joints: list, required
            list of joint ids

    return
    ----------
        Return all maximum forces as a list

    """
    max_forces = []
    for joint in joints:
        forces = get_max_force(body, joint)
        max_forces.append(forces)

    return max_forces


def get_fkine_position(body, link_idx, compute_forward_kinematics=False):
    """
    Return the position of a specific link

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        link_idx: int, required
            link id of a joint
        compute_forward_kinematics: bool, optional
            if set to 1 (or True), the Cartesian world position/orientation will be recomputed
            using forward kinematics
    return
    ----------
        Return link-position as a list
    """
    if compute_forward_kinematics:
        position = p.getLinkState(body,
                                  link_idx,
                                  computeForwardKinematics=compute_forward_kinematics,
                                  physicsClientId=CLIENT)[0]
    else:
        position = p.getLinkState(body, link_idx, physicsClientId=CLIENT)[0]

    return position


def get_fkine_orientation(body, link_idx, compute_forward_kinematics=False):
    """
    Return the orientation of a specific link

    Parameters
    ----------
        body: int, required
            body unique id, as returned by loadURDF etc
        link_idx: int, required
            link id of a joint
        compute_forward_kinematics: bool, optional
            if set to 1 (or True), the Cartesian world position/orientation will be recomputed
            using forward kinematics
    return
    ----------
        Return link-orientation as a list in quaternion angles
    """
    if compute_forward_kinematics:
        orientation = p.getLinkState(body,
                                     link_idx,
                                     computeForwardKinematics=compute_forward_kinematics,
                                     physicsClientId=CLIENT)[1]
    else:
        orientation = p.getLinkState(body, link_idx, physicsClientId=CLIENT)[1]

    return orientation


def set_time_step(t):
    """
    each time you call $stepSimulation$ the timestep will proceed with timestep
    Parameters
    ----------
        t: float, required
            Time for updating simulation step

    """
    p.setTimeStep(t, physicsClientId=CLIENT)
