from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from multiprocessing.context import Process
from matplotlib.figure import Figure
from multiprocessing import Queue
import matplotlib.pyplot as plt
from collections import deque
import multiprocessing as mp
import tkinter as tk
import numpy as np
import threading
import time


def save_list(data, file_name):
    """
    Save list in a CSV file

    Parameters
    ----------
        data: list, required
            Data as a list
        file_name: str, required
            Path of the file
    """
    import csv
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def save_numpy_array(data, file_name):
    """
    Save numpy array

    Parameters
    ----------
        data: bool, required
            Data as a numpy array
        file_name: str, required
            Path of the file_name

    """
    with open(file_name, 'wb') as f:
        np.save(f, data)


def load_numpy_array(filename):
    """
    Load numpy array

    Parameters
    ----------
        filename: str, required
            Path of the file_name

    return
    ----------
        Return tuple of numpy array, or a single numpy array

    """
    with open(filename, 'rb') as f:
        return np.load(f, allow_pickle=True)


def convert_list_rad_2_grad(input_list):
    """
    Convert a list of radian values to grad

    Parameters
    ----------
        input_list: list, required
            list of radians

    return
    ----------
        Return a list of grad

    """
    new_list = []
    for value in input_list:
        minimum = rad_2_grad(value[0])
        maximum = rad_2_grad(value[1])
        new_list.append([minimum, maximum])
    return new_list


def convert_list_grad_2_rad(input_list):
    """
    Convert a list of grad values to rad

    Parameters
    ----------
        input_list: list, required
            list of grad

    return
    ----------
        Return a list of radians

    """
    new_list = []
    for value in input_list:
        minimum = grad_2_rad(value[0])
        maximum = grad_2_rad(value[1])
        new_list.append([minimum, maximum])
    return new_list


def rad_2_grad(value):
    """
    Convert rad to grad with a numpy array

    Parameters
    ----------
        value: numpy, required
            Vector or matrix of a numpy array

    """
    return value * 180 / np.pi


def grad_2_rad(value):
    """
    Convert grad to rad with a numpy array

    Parameters
    ----------
        value: numpy, required
            Vector or matrix of a numpy array

    """
    return value * np.pi / 180


def is_in_joint_velocity_limits(robot, dq):
    """
    Check if current joint velocity and raise error otherwise

    Parameters
    ----------
        robot: object, required
            created robot object in the beginning of the simulation
        dq: list, required
            List of velocity of the joints (not gripper)
    """
    vel_limits = robot.max_velocity

    for vel, q in zip(vel_limits, dq):
        if q < -vel or q > vel:
            raise Exception("joint is not in velocity limit")


def check_custom_limits(values, min_values, max_values):
    """
    Check limits of a given matrix/vector with costum limits.

    Parameters
    ----------
        values: numpy, required
            Shape of a (n, dof)-numpy array of the trajectory
        min_values: numpy, required
            Minimum values of the costum limits
        max_values: numpy, required
            Maximum values of the costum limits

    return
    ----------
        If in limits, return True

    """
    if np.any((values < min_values) | (values > max_values)):
        # limit validation
        return False
    else:
        return True


def clip_to_limits(values, minimum, maximum):
    """
    Clip numpy array values to a minimum or maximum

    Parameters
    ----------
        values: numpy, required
            Shape of a (n, dof)-numpy array of the trajectory
        minimum: numpy, required
            Minimum values of the costum limits
        maximum: numpy, required
            Maximum values of the costum limits

    return
    ----------
        Return clipped numpy array

    """
    if isinstance(values, np.ndarray):
        return np.clip(values, minimum, maximum)
    else:
        return None


def vec_2_matrix(vec, n):
    """
    Convert vector to matrix by coping n-times a row vector

    Parameters
    ----------
        vec: numpy, required
            Row vector that should be copied and tile with a shape of (1, m)
        n: int, required
            Number of rows that should be copied

    return
    ----------
        Return tile matrix
    """
    return np.tile(vec, (n, 1))


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        """
        Buffer for storing data

        Parameters
        ----------
            buffer_size: int, required
                Maximum buffer size of the stored data

        """
        self.count = 0
        self.final_count = 0
        self.buffer_size = buffer_size
        self.buffer = deque()

    def add(self, values):
        """
        Add new values to buffer. 

        Parameters
        ----------
            values: int, required
                new values that should be put into the buffer

        """
        if isinstance(values, np.ndarray):
            for value in values:
                if self.count < self.buffer_size:
                    self.buffer.append(value)
                    self.count += 1
                else:
                    self.buffer.popleft()
                    self.buffer.append(value)
            self.final_count += 1

    def clear(self):
        """
        Clear queue
        """
        self.buffer.clear()
        self.count = 0
        self.final_count = 0


class AppProcess(object):
    app_values = Queue()            # Communication queue from app to simulation
    simulation_values = Queue()     # communication from simulation --> app

    def __init__(self, method, input_values):
        """
        Start new process for the User Interface

        Parameters
        ----------
            method: function, required
                Function that should be paralized
            input_values: tuple, required
                All variables in a tuple handed over to the method function

        """
        self.method = method
        self.input_values = input_values
        self.num_workers = mp.cpu_count()

        self.process = Process(target=self.method, args=(self.input_values, self.app_values, self.simulation_values,))
        self.process.start()


def integration(accel_method, dyn, q, qd, qdd, qdd_t=None, tau=None, dt=1. / 60):
    """
    Euler method with a different name. Because of the limitation of pybullet and the skill of the students,
    a new Euler method has to be implemented. Euler method is one step behind.

    Parameters
    ----------
        accel_method: bool, required
            Function of the acceleration method (has to be programmed from the students)
        dyn: object, required
            Dynamic objects of the robot model
        q: numpy, required
            Current joint positions as a flatten numpy array
        qd: numpy, required
            Current joint velocity as a flatten numpy array
        qdd: numpy, required
            Current joint acceleration as a flatten numpy array
        qdd_t: numpy, required
            New acceleration from the dynamic app as a flatten numpy array
        tau: numpy, required
            new torque from the dynamic app as a flatten numpy array
        dt: int, required
            time step for update

    return
    ----------
        Return new position, velocity and acceleration of one time-step

    """
    q_t = q + qd * dt
    qd_t = qd + qdd * dt

    if isinstance(tau, np.ndarray):
        qdd_t = accel_method(dyn, q_t, qd_t, tau)

    return np.asarray(q_t), np.asarray(qd_t), np.asarray(qdd_t)


class MultiPlot(object):
    def __init__(self, titles, suptitle, supylabel):
        """
        Start plot as a new process. Creates a (1, n)-multiplot. 

        Parameters
        ----------
            titles: list, required
                List of titles strings of the different column plots
            suptitle: str, required
                Name of the suptitle
            supylabel: str, required
                Name of the y axis

        """
        self.fig, self.axs = plt.subplots(1, len(titles), sharey='col', sharex='all')
        self.fig.suptitle(suptitle, fontsize=12)
        self.fig.supylabel(supylabel)

        for i, title in enumerate(titles):
            self.axs[i].set_title(title, fontsize='small')

    def show_plot(self):
        self.fig.show()
        plt.show(block=True)
        time.sleep(5)

    def add_plot(self, input_value, label, color):
        """
        Add new values to all n-subtitles. Iterates new values through a tuple
        and add those values to single subplots with a specific color line. 

        Parameters
        ----------
            input_value: list, required
                List of new input data that should be added to the single subplots
            label: str, required
                label name of the new data that will be added to all subplots
            color: tuple, required
                Color of the line that will be added to all subplots
        """
        for i, value in enumerate(input_value):
            t = np.linspace(0, 1, num=len(value))

            self.axs[i].plot(t, value, label=label, color=color)
            self.axs[i].legend(loc='upper right')


class MultiThreading(object):
    def __init__(self):
        self.num_workers = mp.cpu_count()

    @staticmethod
    def start_threads(threads):
        for t in threads:
            t.start()

    @staticmethod
    def join_threads(threads):
        for t in threads:
            t.join()
    
    def configure_threads(self, methods, args):
        """
        Configures initialised threads and starts threads.

        Parameters
        ----------
            methods: list, required
                list of functions that should be started
            args: tuple, required
                list of tuples of arguments for the called method function
        """
        threads = []
        for method, arg in zip(methods, args):
            threads.append(threading.Thread(target=method, args=(arg,)))

        self.start_threads(threads)
        self.join_threads(threads)


class LivePlotDynamic(tk.Frame):
    def __init__(self, root, dof, limits, data, plot_title=None):
        """
        Plot for robot dynamics integrated in a tkinter app

        Parameters
        ----------
            root: tk, required
                TKinker master root object
            dof: int, required
                Number of joints of the model
            limits: list, required
                Limits of the robot with the shape of (dof, 2)
            data: Queue, required
                All data are stored in a queue
            plot_title: list, optional
                Title name of the plot

        """
        super(LivePlotDynamic, self).__init__()

        self.root = root
        self.dof = dof
        self.limits = limits
        self.dict_data = data

        if plot_title is None:
            self.plot_title = ['Position', 'Velocity', 'Acceleration', 'Torque']
        else:
            self.plot_title = plot_title

        # configure figure in application
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.axs = self.fig.subplots(self.dof, len(self.plot_title))
        self.window_configuration()

        for i, title in enumerate(self.plot_title):
            self.axs[0, i].set_title(title, fontsize='small')
            for j in range(self.dof):
                self.axs[j, i].plot([0], [0], animated=True)

        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(side="top", fill='both', expand=True)
        self.canvas.draw()

    def window_configuration(self):
        # self.fig, self.axs = plt.subplots(self.dof, len(self.plot_title))
        self.fig.suptitle(
            'Tracked joint position, velocity, acceleration and Torque',
            fontsize=12)
        self.fig.supylabel('joints q_i')

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

    def update(self):
        """
        Update plot every 500 ms 
        """
        self.update_plot()
        self.master.after(500, self.update)

    def update_plot(self):
        """
        Update plot with new values and call threads for parallelization
        """
        dyn_data = None

        if not self.dict_data.empty():
            while not self.dict_data.empty():
                dyn_data = self.dict_data.get()
            buffer_size = dyn_data['q'].buffer_size
            final_count = dyn_data['q'].final_count

            args = []
            for i in range(len(self.plot_title)):
                if i == 0:
                    data = np.array(dyn_data['q'].buffer)
                    args.append(
                        (i, data, self.limits['q'], buffer_size, final_count))
                elif i == 1:
                    data = np.array(dyn_data['qd'].buffer)
                    args.append(
                        (i, data, self.limits['qd'], buffer_size, final_count))

                elif i == 2:
                    data = np.array(dyn_data['qd'].buffer)
                    args.append((i, data, self.limits['qdd'], buffer_size,
                                 final_count))
                elif i == 3:
                    data = np.array(dyn_data['tau'].buffer)
                    args.append((i, data, self.limits['tau'], buffer_size,
                                 final_count))

            thread = MultiThreading()
            thread.configure_threads([self.sub_plots] * 4, args=args)

            self.canvas.draw()

    def sub_plots(self, args):
        """
        update suplots
        """
        i, data, limit, buffer_size, final_count = args
        if isinstance(data, np.ndarray):
            for j in range(self.dof):
                self.axs[j, i].clear()
                self.axs[0, i].set_title(self.plot_title[i], fontsize='small')

                if i == 0:
                    min_limit, max_limit = limit[j]
                    self.axs[j, i].set_ylim([min_limit - 1, max_limit + 1])

                    if final_count >= buffer_size:
                        steps = np.linspace((final_count - buffer_size) * 0.5,
                                            final_count * .5, len(data))
                        # np.arange(final_count - buffer_size, final_count)
                        self.axs[j, i].plot(steps, data[:, j], linewidth=2,
                                            color='r')
                    else:
                        steps = np.linspace(0, len(data) * .5, len(data))
                        self.axs[j, i].plot(steps, data[:, j], linewidth=2,
                                            color='r')
                else:
                    self.axs[j, i].set_ylim([-limit[j] - 1, limit[j] + 1])
                    if final_count >= buffer_size:
                        steps = np.linspace((final_count - buffer_size) * .5,
                                            final_count * .5, len(data))
                        self.axs[j, i].plot(steps, data[:, j], linewidth=2,
                                            color='r')
                    else:
                        steps = np.linspace(0, len(data) * .5, len(data))
                        self.axs[j, i].plot(steps, data[:, j], linewidth=2,
                                            color='r')


def run_plot(dof, limits, value_queue):
    """
    Run robot dynamic plot in a tkinter app

    Parameters
        ----------
            dof: int, required
                Number of joints of the model
            limits: list, required
                Limits of the model
            value_queue: tuple, required
                Queue for data transfer and communication with pybullet and tkinter app
    """
    root = tk.Tk(className='dynamic-plot')
    app = LivePlotDynamic(root, dof, limits, value_queue)
    app.root.after(500, app.update)
    app.mainloop()
