from Simulation.utils.pybullet_tools.pybullet_utils import connect, disconnect, get_all_joint_position, \
    set_joint_positions
from Simulation.Robots.robot_primitives import Dynamic, InverseKinematic, MotionControl, PathPlanning
from Simulation.Robots.robot_utils import QR, RobotTCPControl
from Simulation.Enviroment.build_scenario import BuildWorld
from Simulation.utils.utils import AppProcess, run_plot
from Simulation.Tests.ex2_tests import test_workspace
from Simulation.UI.robot_planning import run_app
from Tasks.ex2_tasks import initialize_workspace
from Simulation.Tests.ex3_tests import test_rne
from Simulation.Robots.robot import RobotSetup

from Tasks.ex4_tasks import visualization
from Tasks.ex3_tasks import rne

from multiprocessing.context import Process
from multiprocessing import Queue
import numpy as np
import sys


def main():
    connect()
    scn = BuildWorld()
    robot = RobotSetup(scn.robot, "Panda", file_name=scn.robot_path)

    workspace = test_workspace(initialize_workspace(robot))
    app_process = AppProcess(run_app, input_values=workspace)

    if visualization:
        dict_plot_data = Queue()
        limits = {
            'q': robot.joint_limits,
            'qd': robot.max_velocity,
            'qdd': robot.max_acceleration,
            'tau': robot.max_forces
        }
        plot_process = Process(target=run_plot, args=(len(robot.movable_joints), limits, dict_plot_data))
        plot_process.start()

    dyn = Dynamic(robot)
    tcp_control = RobotTCPControl(robot, workspace)
    ik_solver = InverseKinematic(robot, scn.all_bodies[0])
    planning = PathPlanning(robot, scn.all_bodies[0], visualization=visualization)

    # test rne again
    rne_result = test_rne(rne, dyn)
    
    while True:
        if not app_process.app_values.empty():
            control_values = app_process.app_values.get()
            action, parameters = control_values

            if action == 'tcp_control':
                tcp_control.update(parameters)

            elif action == "add_tcp_position":
                tcp_position, tcp_orientation = parameters[:3], parameters[3:]
                config = ik_solver.search_ik((tcp_position, tcp_orientation, None))

                if config[2]:
                    # send found trajectory message to app
                    app_process.simulation_values.put([True])
                    # draw new tcp position in pybullet
                    tcp_control.add_new_tcp_target(parameters)
                else:
                    # send false trajectory message to app
                    app_process.simulation_values.put([False])

            elif action == "delete_tcp_position":
                tcp_control.delete_single_tcp_target(parameters)

            elif action == "update_tcp_plan":
                tcp_control.swap_tcp_target(parameters)

            elif action == 'reset':
                # delete all add tcp targets
                tcp_control.delete_tcp_targets()

                # set joint position to QR
                set_joint_positions(robot.body, robot.movable_joints, QR)
                
                # reset planning 
                planning.clear_variables()
                planning.clear_trajectory_lines()

            elif action == 'execute_trajectory':
                if not rne_result:
                    print("\n\nDo Exercise 3. RNE implementation not correct\n\n")
                    disconnect()
                    sys.exit()
                
                # reset simulation if after first execution new tcp position in app was added
                current_position = np.array(get_all_joint_position(robot.body))[:robot.dof]
                if not np.array_equal(current_position, QR):
                    set_joint_positions(robot.body, robot.movable_joints, QR)

                    # reset planning 
                    planning.clear_variables()

                tcp_positions = tcp_control.tcp_position
                tcp_orientations = tcp_control.tcp_orientation

                execution_time, execution_pattern = parameters
                if isinstance(tcp_positions, np.ndarray):
                    found_path = planning.plan_execution_path(tcp_positions, tcp_orientations, 
                                                              execution_pattern, execution_time)

                    if found_path:
                        motion = MotionControl(robot, (planning.q, planning.qd, planning.qdd), scn.all_bodies[0], False,
                                               execution_times=execution_time, buffer_size=len(planning.q))
                        motion.execute_motion_control()

                        if visualization:
                            dyn_dict = {
                                "q"  : motion.track_position,
                                "qd" : motion.track_velocity,
                                "qdd": motion.track_acceleration,
                                "tau": motion.track_forces
                            }
                            dict_plot_data.put(dyn_dict)
                    else:
                        print("\n\nDo Exercise 4\n\n")


if __name__ == '__main__':
    main()
