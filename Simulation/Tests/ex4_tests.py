from Simulation.utils.utils import load_numpy_array, vec_2_matrix

import numpy as np
import warnings

coefficient_sol = load_numpy_array("Simulation/Data/coefficient_sol.npy")
q, qd, qdd = load_numpy_array("Simulation/Data/test_trajectory.npy")

t = (0, 5)
steps = 60
ti = np.linspace(t[1], t[0], num=(t[1] - t[0]) * steps)
ti = ti.reshape((len(ti), 1))

# transform single vector into matrix with a specific time interval
length = len(ti)
a_0 = vec_2_matrix(coefficient_sol[0], length)
a_1 = vec_2_matrix(coefficient_sol[1], length)
a_2 = vec_2_matrix(coefficient_sol[2], length)
a_3 = vec_2_matrix(coefficient_sol[3], length)
a_4 = vec_2_matrix(coefficient_sol[4], length)
a_5 = vec_2_matrix(coefficient_sol[5], length)


def test_coefficients(coefficient_method):
    """
    Test coefficients for trajectory tasks

    Parameters
    ----------
        coefficient_method: method, required
            Method for calculating trajectory coefficient
    return
    ----------
        Return boolean for correct implementing of the coefficient tasks
    """
    a0, a1, a2, a3, a4, a5 = coefficient_method(q, qd, qdd, t)

    if isinstance(a0, np.ndarray) and isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray) and \
            isinstance(a3, np.ndarray) and isinstance(a4, np.ndarray) and isinstance(a5, np.ndarray):
        if not np.allclose(a0, coefficient_sol[0], atol=1e-14):
            warnings.warn("ERROR: Invalid coefficient implementation")
            return False

        if not np.allclose(a1, coefficient_sol[1], atol=1e-14):
            warnings.warn("ERROR: Invalid coefficient implementation")
            return False

        if not np.allclose(a2, coefficient_sol[2], atol=1e-14):
            warnings.warn("ERROR: Invalid coefficient implementation")
            return False

        if not np.allclose(a3, coefficient_sol[3], atol=1e-14):
            warnings.warn("ERROR: Invalid coefficient implementation")
            return False

        if not np.allclose(a4, coefficient_sol[4], atol=1e-14):
            warnings.warn("ERROR: Invalid coefficient implementation")
            return False

        if not np.allclose(a5, coefficient_sol[5], atol=1e-14):
            warnings.warn("ERROR: Invalid coefficient implementation")
            return False

        # Test succesful
        return True
    else:
        return False


def test_q_t(q_t_method):
    """
    Test joint position for trajectory tasks

    Parameters
    ----------
        q_t_method: method, required
            Method for calculating trajectory positions
    return
    ----------
        Return boolean for correct implementing of the positions trajectory tasks
    """
    qt_sol = load_numpy_array("Simulation/Data/qt_sol.npy")
    qt = q_t_method(a_0, a_1, a_2, a_3, a_4, a_5, ti)

    if isinstance(qt, np.ndarray):
        if not np.allclose(qt, qt_sol, atol=1e-14):
            warnings.warn("ERROR: Invalid q_t implementation")
            return False
        else:
            return True
    else:
        return False


def test_qd_t(qd_t_method):
    """
    Test velocity for trajectory tasks

    Parameters
    ----------
        qd_t_method: method, required
            Method for calculating trajectory velocity
    return
    ----------
        Return boolean for correct implementing of the velocity trajectory tasks
    """
    qdt_sol = load_numpy_array("Simulation/Data/qdt_sol.npy")
    qdt = qd_t_method(a_1, a_2, a_3, a_4, a_5, ti)

    if isinstance(qdt, np.ndarray):
        if not np.allclose(qdt, qdt_sol, atol=1e-14):
            warnings.warn("ERROR: Invalid qd_t implementation")
            return False
        else:
            return True
    else:
        return False


def test_qdd_t(qdd_t_method):
    """
    Test acceleration for trajectory tasks

    Parameters
    ----------
        qdd_t_method: method, required
            Method for calculating trajectory acceleration
    return
    ----------
        Return boolean for correct implementing of the acceleration tasks
    """
    qddt_sol = load_numpy_array("Simulation/Data/qddt_sol.npy")
    qddt = qdd_t_method(a_2, a_3, a_4, a_5, ti)

    if isinstance(qddt, np.ndarray):
        if not np.allclose(qddt, qddt_sol, atol=1e-14):
            warnings.warn("ERROR: Invalid qdd_t implementation")
            return False
        else:
            return True
    else:
        return False
