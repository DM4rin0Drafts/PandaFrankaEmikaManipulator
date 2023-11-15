from Simulation.utils.pybullet_tools.pybullet_utils import connect, get_all_joint_position, get_limb_positions
from Simulation.Robots.robot_utils import CoordinateSystemControl, RobotTCPControl, UserDebugControl
from Simulation.Tests.ex2_tests import test_inverse_kinematic, test_workspace
from Simulation.Robots.robot_primitives import InverseKinematic
from Simulation.Enviroment.build_scenario import BuildWorld
from Simulation.UI.robot_control import run_app
from Simulation.Robots.robot import RobotSetup
from Simulation.utils.utils import AppProcess

from Tasks.ex2_tasks import initialize_workspace, inverse_kinematic, set_joints, visualization

import numpy as np


def main():
	connect()
	scn = BuildWorld()
	robot = RobotSetup(scn.robot, "Panda", file_name=scn.robot_path)

	# Setup Simulation
	workspace = test_workspace(initialize_workspace(robot))
	joint_config = get_limb_positions(robot.body)
	app_process = AppProcess(run_app, input_values=(joint_config, robot.joint_limits, workspace))

	joint_control = UserDebugControl(robot.body, robot.joints)
	coordinate_control = CoordinateSystemControl(robot, visualization=visualization)

	ik_solver = InverseKinematic(robot, scn.all_bodies[0])
	tcp_control = RobotTCPControl(robot, workspace)

	while True:
		if not app_process.app_values.empty():
			control_values = app_process.app_values.get()

			workspace_parameters = control_values[1][:6]
			joint_parameters = control_values[1][6:]
			tcp_control.update(workspace_parameters)

			if control_values[0] == 'fkine':
				# update slider configuration in simulation
				joint_control.update_robot_joints(joint_parameters)

			elif control_values[0] == 'ikine':
				# search ik
				ik_search_mode = control_values[2]
				position, orientation = workspace_parameters[:3], workspace_parameters[3:]

				joint_config = test_inverse_kinematic(
						robot,
						inverse_kinematic(ik_solver, (position, orientation, None), ik_search_mode))

				if isinstance(joint_parameters, list):
					set_joints(robot, joint_parameters)
					joint_parameters = joint_config
				app_process.simulation_values.put(joint_parameters)

			elif control_values[0] == 'tcp_control':
				# update tcp position in simulation
				joint_values = control_values[1]
				workspace_config = np.concatenate((ik_solver.compute_tcp_position(joint_values, True),
												   ik_solver.compute_tcp_orientation(joint_values)))

				app_process.simulation_values.put(workspace_config)
				tcp_control.update(workspace_config)
			coordinate_control.update(joint_parameters[:7])

		coordinate_control.update(get_all_joint_position(robot.body)[:robot.dof])


if __name__ == '__main__':
	main()
