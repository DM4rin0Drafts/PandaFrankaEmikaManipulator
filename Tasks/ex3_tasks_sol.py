import numpy as np

render = False


def rne(dyn, q, qd, qdd, gravity=None):
    """
    Calculates the forces with the reclusive newton euler method

    Parameters
    ----------
        dyn: object, required
            Object description of the dynamic robot model
        q: numpy, required
            Joint position as a flatten numpy array
        qd: numpy, required
            Joint velocity as a flatten numpy array
        qdd: numpy, required
            Joint acceleration as a flatten numpy array
        gravity: list, optional
            Set a specific gravitational force. If None, it sets a default gravity force (9.81)
    return
    ----------
        Return the forces as a flatten numpy array
    """
    for j in range(dyn.dof):
        dyn.R[j] = dyn.robot.model[j+1].A(q[j]).A[:3, :3].T
        dyn.t[j] = dyn.robot.model[j+1].A(q[j]).A[:3, -1]
    
    if gravity is None:
        N, F = dyn.forward_recursion(qd, qdd)
    else:
        N, F = dyn.forward_recursion(qd, qdd, gravity)

    tau = dyn.backward_recursion(qd, qdd, N, F)
    return np.asarray(tau)


def accel(dyn, q, qd, torque):
    """
    Calculates the acceleration

    Parameters
    ----------
        dyn: Dynamic, required
            Object description of the dynamic robot model
        q: numpy, required
            Joint position as a flatten numpy array
        qd: numpy, required
            Joint velocity as a flatten numpy array
        torque: numpy, required
            Torques as a as a flatten numpy array
    return
    ----------
        Return the acceleration as a flatten numpy array
    """
    for j in range(dyn.dof):
        dyn.R[j] = np.linalg.pinv(dyn.robot.model[j + 1].A(q[j]).A)[:3, :3]
        dyn.t[j] = dyn.robot.model[j + 1].A(q[j]).A[:3, -1]

    qdI = np.zeros((dyn.dof, dyn.dof))
    qddI = np.eye(dyn.dof, dtype=np.float64)
    
    M = np.zeros((dyn.dof, dyn.dof))
    for p, (qd_k, qdd_k) in enumerate(zip(qdI, qddI)):
        N, F =  dyn.forward_recursion(qd_k, qdd_k, [0, 0, 0])  
        M[p, :] = dyn.backward_recursion(qd_k, qdd_k, N, F)

    tauk = rne(dyn, q, qd, np.zeros(dyn.dof))
    qdd = np.linalg.solve(M,  (torque - tauk)[..., np.newaxis])

    return np.asarray(qdd).flatten()


def euler_step(dyn, q, qd, qdd, tau, dt=1./60):
    """
    Calculate one euler step with a specific time step

    Parameters
    ----------
        dyn: object, required
            Object description of the dynamic robot model
        q: numpy, required
            Joint position as a flatten numpy array
        qd: numpy, required
            Joint velocity as a flatten numpy array
        qdd: numpy, required
            Joint acceleration as a flatten numpy array
        tau: numpy, required
            Torques as a as a flatten numpy array
        dt: float, optional
            Time step for the current epoch
    return
    ----------
        Return the next joint position, joint velocity and joint acceleration as flatten numpy arrays
    """
    q_t = q + qd * dt
    qd_t = qd + qdd * dt
    qdd_t = accel(dyn, q_t, qd_t, tau)
        
    return np.asarray(q_t), np.asarray(qd_t), np.asarray(qdd_t)


def gravload(dyn, q):
    """
    Calculates the force compensation for a resting robot model

    Parameters
    ----------
        dyn: object, required
            Object description of the dynamic robot model
        q: numpy, required
            Joint position as a flatten numpy array
    """
    gravity = dyn.a_grav

    z = np.zeros(dyn.dof)     
    taug = rne(dyn, q, z, z, gravity=gravity)

    print("tau: ", taug)
