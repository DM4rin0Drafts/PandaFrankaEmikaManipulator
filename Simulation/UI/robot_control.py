from Simulation.utils.utils import rad_2_grad

from tkinter import ttk
import tkinter as tk
import threading


class RobotControlApp(tk.Frame):
    joint_scales = []
    joint_labels = []

    workspace_scales = []
    workspace_labels = []

    def __init__(self, master, input_values, value_queue, simulation_queue):
        super(RobotControlApp, self).__init__()
        self.master = master
        self.joint_config = input_values[0]
        self.limits = input_values[1]
        self.workspace = input_values[2]
        self.input_values = input_values
        self.value_queue = value_queue
        self.simulation_queue = simulation_queue

        self.robot_control = ttk.Notebook(self.master, height=600, width=450)
        self.workspace_tab = tk.Frame(self.robot_control)
        self.joint_tab = tk.Frame(self.robot_control)

        self.master.geometry("450x600")
        self.master.minsize(450, 600)
        self.master.maxsize(450, 600)

        self.robot_control.add(self.workspace_tab, text='Workspace')
        self.robot_control.add(self.joint_tab, text='Joint Space')
        self.robot_control.pack(expand=1, fill="both")

        self.ik_button = tk.Button(self.workspace_tab,
                                   text="Search Inverse Kinematic",
                                   command=lambda: threading.Thread(
                                       target=self.ik_update).start(),
                                   height=5, width=27)
        self.new_ik_button = tk.Button(self.workspace_tab,
                                       text="Search New Inverse Kinematic",
                                       command=lambda: threading.Thread(
                                           target=self.new_ik_update).start(),
                                       height=5,
                                       width=27)
        self.fk_button = tk.Button(self.joint_tab, text="Forward Kinematic",
                                   command=lambda: threading.Thread(
                                       target=self.fk_update).start(),
                                   height=5, width=27)

        self.fk_button.place(x=125, y=475)
        self.ik_button.place(x=15, y=475)
        self.new_ik_button.place(x=230, y=475)

        self.workspace_control()
        self.joint_control()

    def workspace_control(self):
        for lines in self.workspace:
            minimum, maximum = lines[1], lines[2]
            if minimum == maximum:
                label_workspace_notice = ttk.Label(self.workspace_tab,
                                                   font=('Helvatical bold', 9),
                                                   text="Sliders disabled. Check programming task a)")
                label_workspace_notice.place(x=100, y=375)
                label_workspace_notice.configure(foreground="red")
                break

        y = 30
        name_y = 0

        for i in range(len(self.workspace)):
            workspace_name, min_workspace, max_workspace, start_position = self.workspace[i]

            ttk.Label(self.workspace_tab, font=('Helvatical bold', 12),
                      text=workspace_name + ":").place(x=20,
                                                       y=name_y)

            self.workspace_scales.append(ttk.Scale(self.workspace_tab,
                                                   from_=min_workspace,
                                                   to=max_workspace,
                                                   length=300,
                                                   value=start_position,
                                                   orient=tk.HORIZONTAL,
                                                   command=self.update_slider))
            self.workspace_scales[i].place(x=80, y=y)

            if i >= 3:
                unit = '°'
                min_value = str(round(rad_2_grad(min_workspace), 2))
                max_value = str(round(rad_2_grad(max_workspace), 2))
                start_value = str(round(rad_2_grad(start_position), 2))
            else:
                unit = 'm'
                min_value = str(round(min_workspace, 2))
                max_value = str(round(max_workspace, 2))
                start_value = str(round(start_position, 2))

            ttk.Label(self.workspace_tab, text=min_value + unit).place(x=20, y=y)
            ttk.Label(self.workspace_tab, text=max_value + unit).place(x=390, y=y)
            self.workspace_labels.append(tk.Label(self.workspace_tab,
                                                  text=start_value + unit))
            self.workspace_labels[i].place(x=225, y=name_y)

            y = y + 60
            name_y = name_y + 60

    def joint_control(self):
        y = 30
        name_y = 0
        for i in range(len(self.joint_config)):
            min_limit, max_limit = self.limits[i]
            if len(self.joint_config) - 1 == i:
                title = "Gripper"
                min_label_text = str(round(min_limit, 2)) + "m"
                max_label_text = str(round(max_limit, 2)) + "m"
                current_position_text = str(
                    round(self.joint_config[i], 2)) + "m"
            else:
                title = "Joint " + str(i + 1)
                min_label_text = str(round(rad_2_grad(min_limit), 2)) + "°"
                max_label_text = str(round(rad_2_grad(max_limit), 2)) + "°"
                current_position_text = str(
                    round(rad_2_grad(self.joint_config[i]), 2)) + "°"

            ttk.Label(self.joint_tab, font=('Helvatical bold', 12),
                      text=title).place(x=20, y=name_y)
            ttk.Label(self.joint_tab, text=min_label_text).place(x=20, y=y)
            ttk.Label(self.joint_tab, text=max_label_text).place(x=390, y=y)

            self.joint_scales.append(ttk.Scale(self.joint_tab,
                                               from_=min_limit,
                                               to=max_limit,
                                               length=300,
                                               value=self.joint_config[i],
                                               orient=tk.HORIZONTAL,
                                               command=self.update_slider))
            self.joint_scales[i].place(x=80, y=y)
            self.joint_labels.append(tk.Label(self.joint_tab,
                                              text=current_position_text))
            self.joint_labels[i].place(x=225, y=name_y)

            y = y + 60
            name_y = name_y + 60

    def update_slider(self, val):
        self.slider_values = []
        for idx, slider in enumerate(self.workspace_scales):
            self.slider_values.append(slider.get())

            if idx >= 3:
                self.workspace_labels[idx]['text'] = str(
                    (round(rad_2_grad(slider.get()), 2))) + "°"
            else:
                self.workspace_labels[idx]['text'] = str(
                    round(slider.get(), 2)) + "m"

        for idx, slider in enumerate(self.joint_scales):
            self.slider_values.append(slider.get())
            if idx == len(self.joint_config) - 1:
                self.joint_labels[idx]['text'] = str(
                    round(slider.get(), 2)) + "m"
            else:
                self.joint_labels[idx]['text'] = str(
                    round(rad_2_grad(slider.get()), 2)) + "°"

        # put current slider values in queue
        if not self.value_queue.empty():
            _ = self.value_queue.get()
            self.value_queue.put(['fkine', self.slider_values])
        else:
            self.value_queue.put(['fkine', self.slider_values])

        # print("slider information: ", self.slider_values)

    @staticmethod
    def set_sliders(scales, values):
        for slider, value in zip(scales, values):
            slider.set(value)

    def read_sliders(self):
        values = []
        for slider in self.workspace_scales:
            values.append(slider.get())

        for slider in self.joint_scales:
            values.append(slider.get())

        return values

    def ik_update(self):
        values = self.read_sliders()  # take current slider values

        # if self.value_queue is False, then joint control, if true search ik
        self.value_queue.put(["ikine", values, True])

        while self.simulation_queue.empty():
            continue
            # waiting for found inverse kinematic ans set it
        else:
            joint_values = self.simulation_queue.get()

        # set joint values in app
        self.set_sliders(self.joint_scales, joint_values)

    def new_ik_update(self):
        values = self.read_sliders()  # take current slider values

        # if self.value_queue is False, then joint control, if true search ik
        self.value_queue.put(["ikine", values, False])

        while self.simulation_queue.empty():
            continue
            # waiting for found inverse kinematic ans set it
        else:
            joint_values = self.simulation_queue.get()

        # set joint values in app
        self.set_sliders(self.joint_scales, joint_values)

    def fk_update(self):
        values = self.read_sliders()[6:][:-1]
        self.value_queue.put(["tcp_control", values])

        while self.simulation_queue.empty():
            continue
        else:
            workspace_values = self.simulation_queue.get()

        self.set_sliders(self.workspace_scales, workspace_values)


def run_app(input_values, value_queue, simulation_queue):
    root = tk.Tk(className='robot-control')
    robot_control_app = RobotControlApp(root,
                                        input_values,
                                        value_queue,
                                        simulation_queue)
    robot_control_app.mainloop()
