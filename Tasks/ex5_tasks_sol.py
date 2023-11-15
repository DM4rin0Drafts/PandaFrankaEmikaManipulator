from Simulation.utils.utils import clip_to_limits
from Tasks.ex3_tasks_sol import accel, rne
import numpy as np


T_B = 110


def unrestrained_movement(dyn, robot, q, qd, qdd, h=1./240):
    """
    Calculates the trajectory of a given trajectory with an euler step

    Parameters
    ----------
        dyn: Dynamic, required
            Object description of the dynamic robot model
        robot: RobotSetup, required
            Object description of the robot model with all important information
        q: numpy, required
            Trajectory for the joint position as a (n, dof) numpy array
        qd: numpy, required
            Trajectory for the joint velocity as a (n, dof) numpy array
        qdd: numpy, required
            Trajectory for the joint acceleration as a (n, dof) numpy array
        h: float, optional
            Time step
    return
    ----------
        Return new trajectory for joint position, joint velocity, joint acceleration and the forces 
        as a numpy array with a shape of (n, dof)
    """
    dof = robot.dof

    forces = []
    for i in range(1, len(q)):
        tau = rne(dyn, q[i-1], qd[i-1], qdd[i])
        forces.append(tau)

        qddt = accel(dyn, q[i-1], qd[i-1], tau)
        qdd[i] = clip_to_limits(qddt, -robot.max_acceleration[:dof], robot.max_acceleration[:dof])

        qdt = qd[i-1] + qdd[i] * h
        qd[i] = clip_to_limits(qdt, -robot.max_velocity[:dof], robot.max_velocity[:dof])

        qt = q[i-1] + qd[i] * h
        q[i] = clip_to_limits(qt, robot.joint_limits[:dof, 0], robot.joint_limits[:dof, 1])

    return q, qd, qdd, np.asarray(forces)


def restrained_movement(dyn, robot, q, qd, qdd, h=1./240.):
    """
    Calculates a new trajectory of a given trajectory with an euler step. Braking and slowing down movement

    Parameters
    ----------
        dyn: object, required
            Description of the dynamic robot model
        robot: object, required
            Description of the robot model with all important information
        q: numpy, required
            A trajectory for the joint position as a (n, dof) numpy array
        qd: numpy, required
            A trajectory for the joint velocity as a (n, dof) numpy array
        qdd: numpy, required
            A trajectory for the joint acceleration as a (n, dof) numpy array
        h: float, optional
            Time step
    return
    ----------
        Return new trajectory for joint position, joint velocity, joint acceleration and the forces 
        as a numpy array with a shape of (n, dof)
    """
    dof = robot.dof 

    forces = []
    for i in range(1, len(q)):
        tau = rne(dyn, q[i-1], qd[i-1], qdd[i])
        forces.append(tau)

    for i in range(T_B, len(q)):
        if i >= T_B:
            tau = forces[i-1]
            tau[0] = -2.5*np.sign(qd[i-1][0]) 

            qdd[i][0] = accel(dyn, q[i-1], qd[i-1], tau)[0] 
            qdd[i] = clip_to_limits(qdd[i], -robot.max_acceleration[:dof], robot.max_acceleration[:dof])

            qd[i][0] = qd[i-1][0] + qdd[i][0] * h
            qd[i] = clip_to_limits(qd[i], -robot.max_velocity[:dof], robot.max_velocity[:dof])

            q[i][0] = q[i-1][0] + qd[i][0] * h
            q[i] = clip_to_limits(q[i], robot.joint_limits[:dof, 0], robot.joint_limits[:dof, 1])

    return q, qd, qdd, np.asarray(forces)
