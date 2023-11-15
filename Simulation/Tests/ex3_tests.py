from Simulation.utils.utils import load_numpy_array
from Simulation.Robots.robot_utils import QR

import numpy as np
import warnings


true_force = load_numpy_array("Simulation/Data/force_solution.npy")
true_accel = load_numpy_array("Simulation/Data/accel_solution.npy")


def test_rne(rne_method, dyn, q=None, qd=None, qdd=None):
    """
    Test return values of reclusive euler method if rne method is right implemented

    Parameters
    ----------
        rne_method: method, required
            Object of the robot description
        dyn: object, required
            Description the dynamics of the robot
        q: numpy, optional, flatten shape
            Joint position of the robot
        qd: numpy, optional, flatten shape
            joint velocity of the robot
        qdd: numpy, optional, flatten shape
            joint acceleration of the robot
    return
    ----------
        Return default values or the true force as a numpy array
    """
    if qdd is None:
        qdd = [0.0] * 7
    if qd is None:
        qd = [0.0] * 7
    if q is None:
        q = QR
    test_force = rne_method(dyn, np.array(q), np.array(qd), np.array(qdd))

    if isinstance(test_force, np.ndarray):
        if test_force.shape != (7,):
            warnings.warn("ERROR: invalid rne return shape")
            return False
        
        if len(test_force) != 7:
            warnings.warn("ERROR: invalid rne return length")
            return False

        if not np.allclose(test_force, true_force, atol=1e-14):
            warnings.warn("ERROR: invalid rne implementation")
            return False
        
        return True
    else:
        warnings.warn("ERROR: invalid rne implementation")
        return False


def test_accel(accel_method, dyn, q=None, qd=None):
    """
    Test return values of acceleration if it right implemented

    Parameters
    ----------
        accel_method: method, required
            Object of the robot description
        dyn: object, required
            Description the dynamics of the robot
        q: numpy, optional
            joint position of the robot
        qd: numpy, optional
            joint velocity of the robot
    return
    ----------
        Return default values or acceleration as a numpy flatten array
    """
    if qd is None:
        qd = [0.0] * 7
    if q is None:
        q = QR
    test_accel = accel_method(dyn, np.array(q), np.array(qd), np.array(true_force))

    if isinstance(test_accel, np.ndarray):
        if test_accel.shape != (7,):
            warnings.warn("ERROR: invalid rne return shape")
            return False
    
        if len(test_accel) != 7:
            warnings.warn("ERROR: invalid rne return length")
            return False

        if not np.allclose(test_accel, true_accel, atol=1e-14):
            warnings.warn("ERROR: invalid rne implementation")
            return False

        return True
    else:
        warnings.warn("ERROR: invalid rne implementation")
        return False


def test_euler(euler_method, integration_method, accel_method, dyn):
    """
    Test euler method if it right implemented

    Parameters
    ----------
        euler_method: method, required
            Method of the euler task method
        integration_method: method, required
            Method of the euler task, see documentation
        accel_method: method, required
            Method of the acceleration task method
        dyn: object, required
            Description the dynamics of the robot
    """
    qd = np.array([0.0] * 7)
    q = np.array([0.0, 0.3993, 0.08, -1.6552, 0.1761, 2.6486, 0.0])
    qdd = np.array([0.0]*7)
    tau = true_force + 1.0

    q_t, qd_t, qdd_t = euler_method(dyn, q, qd, qdd, tau)

    if not isinstance(q_t, np.ndarray) and not isinstance(qd_t, np.ndarray) and not isinstance(qdd_t, np.ndarray):
        warnings.warn("ERROR: invalid euler implementation.")
        return False
    else: 
        for i in range(3):
            q_t, qd_t, qdd_t = euler_method(dyn, q, qd, qdd, tau)
            q_t_i, qd_t_i, qdd_t_i = integration_method(accel_method, dyn, q, qd, qdd, tau=tau)

            if not ((q_t == q_t_i).all() and (qd_t == qd_t_i).all() and (qdd_t == qdd_t_i).all()):
                warnings.warn("ERROR: invalid euler implementation")
                return False

            q, qd, qdd = q_t, qd_t, qdd_t
    return True
