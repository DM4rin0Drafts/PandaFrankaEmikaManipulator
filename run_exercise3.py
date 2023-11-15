from Simulation.utils.pybullet_tools.pybullet_utils import connect, get_all_joint_position, set_joint_positions
from Simulation.utils.utils import AppProcess, integration, load_numpy_array, run_plot
from Simulation.Tests.ex3_tests import test_euler, test_rne, test_accel
from Simulation.Robots.robot_primitives import Dynamic, MotionControl
from Tasks.ex3_tasks import euler_step, gravload, rne, accel, render
from Simulation.Enviroment.build_scenario import BuildWorld
from Simulation.UI.robot_dynamic import run_app
from Simulation.Robots.robot_utils import QR
from Simulation.Robots.robot import RobotSetup

from multiprocessing.context import Process
from multiprocessing import Queue
import numpy as np
import pybullet as p


def start_simulation(robot):
	q = np.array(get_all_joint_position(robot.body)[:robot.dof])
	qd = np.zeros(len(robot.movable_joints))
	qdd = np.zeros(len(robot.movable_joints))

	return q, qd, qdd


def main():
	mode = 1
	dict_plot_data = Queue()

	if not render:
		connect(p.DIRECT)
	else:
		connect()

	scn = BuildWorld()
	robot = RobotSetup(scn.robot, "Panda", file_name=scn.robot_path)

	dyn = Dynamic(robot)
	control = MotionControl(robot, (None, None, None), scn.all_bodies[0], True, control_mode=p.POSITION_CONTROL)
	limits = {
		'q': robot.joint_limits[:robot.dof],
		'qd': robot.max_velocity,
		'qdd': robot.max_acceleration,
		'tau': robot.max_forces
	}

	plot_process = Process(target=run_plot, args=(len(robot.movable_joints), limits, dict_plot_data))
	plot_process.start()

	q, qd, qdd = start_simulation(robot)

	torque_values = load_numpy_array("Simulation/Data/force_solution.npy")
	acceleration_values = np.zeros(len(robot.movable_joints))

	# only if euler implementation is correct, then simulation and plot working
	euler_result = test_euler(euler_step, integration, accel, dyn)
	rne_result = test_rne(rne, dyn)
	accel_result = test_accel(accel, dyn)

	# execute gravload task
	gravload(dyn, q)

	app_process = AppProcess(run_app, input_values=(robot.max_forces, robot.max_acceleration, torque_values))
	while True:
		q_t, qd_t, qdd_t = euler_step(dyn, q, qd, qdd, tau=torque_values)

		if euler_result and rne_result and accel_result:
			if not app_process.app_values.empty():
				# get new data from app
				mode, slider_values = app_process.app_values.get()

				torque_values = np.array(slider_values[:len(robot.movable_joints)])
				acceleration_values = np.array(slider_values[len(robot.movable_joints):])

				if mode == "torque":
					if not isinstance(q_t, np.ndarray) and not isinstance(qd_t, np.ndarray) \
							and not isinstance(qdd_t, np.ndarray):
						pass
					else:
						q, qd, qdd = integration(accel, dyn, q, qd, qdd, tau=torque_values)
				elif mode == 'acceleration':
					if not isinstance(q_t, np.ndarray) and not isinstance(qd_t, np.ndarray) \
							and not isinstance(qdd_t, np.ndarray):
						pass
					else:
						q, qd, qdd = integration(accel, dyn, q, qd, qdd, qdd_t=acceleration_values)
				elif mode == 'reset':
					# reset simulation
					set_joint_positions(robot.body, robot.movable_joints, QR)
					q, qd, qdd = start_simulation(robot)
					torque_values = rne(dyn, q, qd, qdd)
					acceleration_values = qdd
			else:
				# use old data to proceed robot in simulation
				if mode == "torque":
					if not isinstance(q_t, np.ndarray) and not isinstance(qd_t, np.ndarray) \
							and not isinstance(qdd_t, np.ndarray):
						pass
					else:
						q, qd, qdd = integration(accel, dyn, q, qd, qdd, tau=torque_values)
				elif mode == 'acceleration':
					if not isinstance(q_t, np.ndarray) and not isinstance(qd_t, np.ndarray) \
							and not isinstance(qdd_t, np.ndarray):
						pass
					else:
						q, qd, qdd = integration(accel, dyn, q, qd, qdd, qdd_t=acceleration_values)

		# check limits
		qdd = np.clip(qdd, -np.asarray(robot.max_acceleration), np.asarray(robot.max_acceleration))

		qd_indices = abs(qd) > np.asarray(robot.max_velocity)
		qd = np.clip(qd, -np.asarray(robot.max_velocity), np.asarray(robot.max_velocity))
		qdd[qd_indices] = 0

		q_indices = (q < robot.joint_limits[:robot.dof, 0]) | (q > robot.joint_limits[:robot.dof, 1])
		q = np.clip(q, robot.joint_limits[:robot.dof, 0], robot.joint_limits[:robot.dof, 1])
		qd[q_indices] = 0
		qdd[q_indices] = 0

		if rne_result and accel_result:
			tau = rne(dyn, q, qd, qdd)
			tau = np.clip(tau, -np.array(robot.max_forces), np.array(robot.max_forces))
			control.execute_dynamic_motion_control(q, qd, qdd, tau)
			app_process.simulation_values.put((tau, qdd))

			dyn_dict = {
				"q"  : control.track_position,
				"qd" : control.track_velocity,
				"qdd": control.track_acceleration,
				"tau": control.track_forces
			}
			dict_plot_data.put(dyn_dict)


if __name__ == '__main__':
	main()
