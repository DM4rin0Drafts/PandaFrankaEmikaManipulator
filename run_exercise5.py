from Simulation.Robots.robot_primitives import Dynamic, MotionControl, PathPlanning, PolynomialTrajectory
from Simulation.utils.pybullet_tools.pybullet_utils import connect, disconnect, get_all_joint_position
from Simulation.utils.utils import MultiPlot, load_numpy_array
from Simulation.Enviroment.build_scenario import BuildWorld
from Simulation.Tests.ex3_tests import test_rne
from Simulation.Robots.robot import RobotSetup

from Tasks.ex3_tasks import rne
from Tasks.ex5_tasks import unrestrained_movement, restrained_movement

import numpy as np
import time
import sys


def main():
    connect(dt=1./60.)
    scn = BuildWorld(True)
    robot = RobotSetup(scn.robot, "Panda", file_name=scn.robot_path)

    plots = MultiPlot(['Position', 'Velocity', 'Acceleration', 'Forces'],
                      'Position, velocity, acceleration and forces',
                      'joints q1')

    q0 = np.array(get_all_joint_position(robot.body)[:robot.dof])
    q1 = load_numpy_array("Simulation/Data/load_trajectory.npy")
    
    dyn = Dynamic(robot, brake_task=True)
    trajectory = PolynomialTrajectory(robot)
    planning = PathPlanning(robot, scn.all_bodies[0])
    
    # polynomial trajectory
    q, qd, qdd = trajectory.compute_trajectory((q0, q1), (np.zeros(7), np.zeros(7)), (np.zeros(7), np.zeros(7)), (0, 1),
                                               steps=240)
    if not isinstance(q, np.ndarray) or not isinstance(qd, np.ndarray) or not isinstance(qdd, np.ndarray):
        print("\n\nDo Exercise 5.\n\n")
        disconnect()
        sys.exit()
        
    planning.draw_trajectory(q=q, local=True)

    # test rne again
    rne_result = test_rne(rne, Dynamic(robot))
    if not rne_result:
        print("\n\nDo Exercise 3. RNE implementation not correct\n\n")
        disconnect()
        sys.exit()

    forces = []
    for i in range(1, len(q)):
        tau = rne(dyn, q[i-1], qd[i-1], qdd[i])
        forces.append(tau)
    forces = np.asarray(forces)
    plots.add_plot((q[:, 0], qd[:, 0], qdd[:, 0], forces[:, 0]),  "trajectory", 'red')
    
    # euler step
    q_u, qd_u, qdd_u, forces_u = unrestrained_movement(dyn, robot, q.copy(), qd.copy(), qdd.copy(), h=1./240.)
    if not np.isnan(q_u).any():
        planning.draw_trajectory(q=q_u, local=True, color=(0, 1, 0))
        plots.add_plot((q_u[:, 0], qd_u[:, 0], qdd_u[:, 0], forces_u[:, 0]), "euler", 'green')

    # brake with euler step
    q_b, qd_b, qdd_b, forces_b = restrained_movement(dyn, robot, q.copy(), qd.copy(), qdd.copy(), h=1./240)
    if not np.isnan(q_b).any():
        planning.draw_trajectory(q=q_b, local=True, color=(0, 0, 1))
        plots.add_plot((q_b[:, 0], qd_b[:, 0], qdd_b[:, 0], forces_b[:, 0]), "brake", 'blue')

    # execute trajectory with brake robot arm
    if isinstance(q_b, np.ndarray) and not np.isnan(q_b).any():
        print("\n\nSuccessful brake. No Collision\n\n")
        control = MotionControl(robot, (q_b, qd_b, qdd_b), scn.all_bodies[0], False, dt=1./240.)
        control.execute_motion_control(tracking=False)  # TODO ohne collision check
    else:
        print("\n\nExecute normal trajectory. Do implementation task.\n\n")

        control = MotionControl(robot, (q, qd, qdd), scn.all_bodies[0], False, dt=1./240.)
        control.execute_motion_control(tracking=False) 
        time.sleep(5)
    
    plots.show_plot()
    disconnect()
    sys.exit()


if __name__ == '__main__':
    main()
