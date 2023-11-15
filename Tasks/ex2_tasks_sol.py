import numpy as np
from Simulation.utils.pybullet_tools.pybullet_utils import all_joints, set_joint_positions

visualization = True


def initialize_workspace(robot):
    """
    Initialize an approximately workspace of the robot model.

    Parameters
    ----------
        robot: RobotSetup, required
            Object description of the robot model with all important information
    return
    ----------
        Return a defined workspace, that can be loaded into the app.
    """
    base_position = robot.base_position
    workspace = [("x", base_position[0] - 1.2,     base_position[0] + 1.2,    0),
                 ("y", base_position[1] - .6,      base_position[1] + .6,     0),
                 ("z",          0,                 base_position[2] + 1.,     1.5),
                 ("rx", -np.pi, np.pi, 0),
                 ("ry", -np.pi, np.pi, 0),
                 ("rz", -np.pi, np.pi, 0)]
    
    return workspace


def inverse_kinematic(ik_solver, input_tuple, search_type):
    """
    Calculates/Search the joint positions for a target positions.

    Parameters
    ----------
        ik_solver: InverseKinematic, required
            Object description of the InverseKinematic
        input_tuple: tuple ,required
            Input tuple of the target-position (numpy, shape=(3,)), target-orientation (numpy, shape(3,)) 
            as euler angles and a list of target bodies in the environment
        search_type: bool, required
            Search a new joint position, when target position is already reached
    return
    ----------
        Return the found joint position
    """
    joint_positions = ik_solver.search_ik(input_tuple, search_type=search_type)[1]

    return joint_positions


def set_joints(robot, joint_positions):
    """
    Set the joint position into the simulation

    Parameters
    ----------
        robot: object, required
            Object description of the robot model with all important information
        joint_positions: list, required
            All movable joint position that should be set
    """
    joints = all_joints(robot.body)[:7]
    set_joint_positions(robot.body, joints, joint_positions)


def Rx(alpha):
    """
    Get connection of a specific simulation. Check connection of the client

    Parameters
    ----------
        alpha: float, required
            Radiant angle of the rotation
    return
    ----------
        Return a 4x4 x-rotation matrix as a numpy array
    """
    rx = [[1, 0, 0, 0],
          [0, np.cos(alpha), -np.sin(alpha), 0],
          [0, np.sin(alpha), np.cos(alpha), 0],
          [0, 0, 0, 1]]
    
    return np.asarray(rx)


def Ry(alpha):
    """
    Get connection of a specific simulation. Check connection of the client

    Parameters
    ----------
        alpha: float, required
            Radiant angle of the rotation
    return
    ----------
        Return a 4x4 y-rotation matrix as a numpy array
    """
    ry = [[np.cos(alpha), 0, np.sin(alpha), 0],
          [0, 1, 0, 0],
          [-np.sin(alpha), 0, np.cos(alpha), 0],
          [0, 0, 0, 1]]

    return np.asarray(ry)


def Rz(alpha):
    """
    Get connection of a specific simulation. Check connection of the client

    Parameters
    ----------
        alpha: float, required
            Radiant angle of the rotation
    return
    ----------
        Return a 4x4 z-rotation matrix as a numpy array
    """
    rz = [[np.cos(alpha), -np.sin(alpha), 0, 0],
          [np.sin(alpha), np.cos(alpha), 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]

    return np.asarray(rz)


def translation(position):
    """
    Compute the translation matrix.

    Parameters
    ----------
        position: list/numpy, required
            Target position as a [X, Y, Z] list or a flatten numpy array
    return
    ----------
        Return a 4x4 translation matrix as a numpy array
    """
    t = np.eye(4)
    t[0:3, -1] = position

    return np.asarray(t)


def compute_transformation(position, orientation):
    """
    Compute the transformation matrix.

    Parameters
    ----------
        position: list/numpy, required
            Target position as a [X, Y, Z] list or a flatten numpy array
        orientation: list/numpy, required
            Target orientation as euler angles. [rX, rY, rZ] as a list or a flatten numpy array 
    return
    ----------
        Return the 4x4 transformation matrix for the end-effector as a numpy array
    """
    T = translation(position) @ Rz(orientation[2]) @ Ry(orientation[1]) @ Rx(orientation[0])

    return np.asarray(T)
