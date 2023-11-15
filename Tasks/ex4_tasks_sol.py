visualization = True


def compute_coefficient(q, qd, qdd, t):
    """
    Calculates the coefficients for a specific trajectory

    Parameters
    ----------
        q: tuple, required
            Start and end joint position as a flatten numpy array
        qd: tuple, required
            Start and end joint velocity as a flatten numpy array
        qdd: tuple, required
            Start and end joint acceleration as a flatten numpy array
        t: tuple, required
            Start and end time as integer values (execution time) with t_0 = 0
    return
    ----------
        Return coefficient values as a numpy array as a shape of (n, dof)
    """
    q_0, q_1 = q
    qd_0, qd_1 = qd
    qdd_0, qdd_1 = qdd
    t_0, t_1 = t

    T = t_1 - t_0
    h = q_1 - q_0  # total displacement

    a_0 = q_0
    a_1 = qd_0
    a_2 = 0.5 * qdd_0
    # alternative: x ** 3 because its numpy
    a_3 = (1 / (2 * pow(T, 3))) * (20 * h - (8 * qd_1 + 12 * qd_0) * T
                                   - (3 * qdd_0 - qdd_1) * pow(T, 2))
    a_4 = (1 / (2 * pow(T, 4))) * (-30 * h + (14 * qd_1 + 16 * qd_0) * T
                                   + (3 * qdd_0 - qdd_1) * pow(T, 2))
    a_5 = (1 / (2 * pow(T, 5))) * (12 * h - 6 * (qd_1 - qd_0) * T
                                   + (qdd_0 - qdd_1) * pow(T, 2))

    return a_0, a_1, a_2, a_3, a_4, a_5


def q_t(a_0, a_1, a_2, a_3, a_4, a_5, ti):
    """
    Calculates the trajectory of the joint position for every time step

    Parameters
    ----------
        a_0: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_1: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_2: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_3: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_4: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_5: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        ti: numpy, required
            Time steps for the trajectory as a flatten numpy array of the lenght of n
    return
    ----------
        Return the joint position as a (n, dof) numpy array
    """
    qt = a_0 + a_1 * ti + a_2 * pow(ti, 2) + a_3 * pow(ti, 3) + a_4 * pow(ti, 4) + a_5 * pow(ti, 5)

    return qt


def qd_t(a_1, a_2, a_3, a_4, a_5, ti):
    """
    Calculates the trajectory of the joint velocity for every time step

    Parameters
    ----------
        a_1: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_2: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_3: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_4: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_5: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        ti: numpy, required
            Time steps for the trajectory as a flatten numpy array of the lenght of n
    return
    ----------
        Return the joint velocity as a (n, dof) numpy array
    """
    qdt = a_1 + 2 * a_2 * ti + 3 * a_3 * pow(ti, 2) + 4 * a_4 * pow(ti, 3) + 5 * a_5 * pow(ti, 4)

    return qdt


def qdd_t(a_2, a_3, a_4, a_5, ti):
    """
    Calculates the trajectory of the joint acceleration for every time step

    Parameters
    ----------
        a_2: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_3: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_4: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        a_5: numpy, required
            Trajectory coefficient as a (n, dof) numpy array
        ti: numpy, required
            Time steps for the trajectory as a flatten numpy array of the lenght of n
    return
    ----------
        Return the joint acceleration as a (n, dof) numpy array
    """
    qddt = 2 * a_2 + 6 * a_3 * ti + 12 * a_4 * pow(ti, 2) + 20 * a_5 * pow(ti, 3)

    return qddt
