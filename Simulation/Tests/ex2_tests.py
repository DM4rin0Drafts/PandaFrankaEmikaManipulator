from Simulation.utils.pybullet_tools.pybullet_utils import get_all_joint_position, get_real_path
from Simulation.utils.utils import load_numpy_array

import numpy as np
import warnings


def load_default_workspace():
    """
    Loading default workspace
    """
    workspace = np.loadtxt('Simulation/Data/default_workspace.npy', dtype=np.object).tolist()
    for i, line in enumerate(workspace):
        for j, value in enumerate(line):
            if j != 0:
                workspace[i][j] = float(value)
    return workspace


def test_workspace(workspace):
    """
    Test workspace if it in correct shape and size for loading values into app

    Parameters
    ----------
        workspace: list, required
            [6x4] list of important workspace parameters
    return
    ----------
        Default workspace or input workspace
    """
    if isinstance(workspace, list):
        if len(workspace) != 6:
            warnings.warn("Check programming task a). Incorrect number of lines. Loading default workspace parameters")
            workspace = load_default_workspace()

        if [len(line) for line in workspace].count(4) != 6:
            warnings.warn("Check programming task a). Incorrect number of columns."
                          " Loading default workspace parameters")
            workspace = load_default_workspace()
    elif workspace is None:
        warnings.warn("Check programming task a). Not implementet yet. Loading default workspace parameters")
        workspace = load_default_workspace()

    return workspace


def test_inverse_kinematic(robot, joint_positions):
    """
    Test return values of inverse kinematic if joint_position is have a specific shape

    Parameters
    ----------
        robot: object, required
            Object of the robot description
        joint_positions: list, required
            list of the current/calculated joint positions
    return
    ----------
        Return default values or joint_positions as a list
    """
    if isinstance(joint_positions, np.ndarray):
        joint_positions = list(joint_positions)

    if joint_positions is None:
        warnings.warn("Check programming task c). Not implemented. Check return value. "
                      "Returning current joint position")
        joint_positions = list(get_all_joint_position(robot.body))[:robot.dof]

    elif len(joint_positions) != robot.dof:
        warnings.warn("Check programming task c). Incorrect joint parameters. Check return value. "
                      "Returning current joint position")
        joint_positions = list(get_all_joint_position(robot.body))[:robot.dof]

    return joint_positions


def test_transformation_matrix(transformation_method):
    """
    Test if the calculated transformation method task is right implemented

    Parameters
    ----------
        transformation_method: method, required
            call the implemented transformation matrix to check the result
    return
    ----------
        Return default transformation matrix or transformation matrix and if the test was successful
    """
    # test transformation matrix
    T_solution = load_numpy_array(get_real_path("/Simulation/Data/default_transformation_matrix.npy"))
    T_test = transformation_method(np.array([0.7656944944600366, 0.05950726127270593, 0.3627181303757686]),
                                   np.array([2.84175502006334, -0.5418033117343002, 0.80363184145752]))

    if isinstance(T_test, np.ndarray):
        if T_test.shape != (4, 4):
            warnings.warn("Check programming task b). Incorrect transformation matrix shape. "
                          "Returning default transformation matrix")
            return False

        if not np.allclose(T_solution, T_test, atol=1e-14):
            warnings.warn("Check programming task b). Incorrect transformation matrix calculation. "
                          "Returning default transformation matrix")
            return False

        # successful
        return True

    warnings.warn("Check programming task b). Result is not correct. Returning default transformation matrix")
    return False
