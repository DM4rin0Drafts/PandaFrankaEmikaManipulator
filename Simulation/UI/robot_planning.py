from Simulation.utils.utils import rad_2_grad

from tkinter import Variable, ttk
import tkinter as tk

from numpy import empty


class DynamicGrid(object):
    targets = []
    positions = []
    orientations = []
    space = []
    execution_time = []
    velocity = []
    trajectory = []
    swap = []
    delete_target = []

    def on_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def __init__(self, tab):
        self.tab = tab
        self.boxes = []

        self.frame = tk.Frame(self.tab, height=350, width=800)
        self.frame.place(x=0, y=0)

        self.canvas = tk.Canvas(self.frame, height=350, width=800)
        self.canvas.place(x=0, y=0)

        self.grid_frame = tk.Frame(self.canvas, height=350, width=800)
        self.grid_frame.bind('<Configure>', self.on_configure)

        self.canvas.create_window(0, 0, window=self.grid_frame)

        self.scrollbar = tk.Scrollbar(self.frame, orient='vertical', command=self.canvas.yview)
        self.scrollbar.place(x=800, rely=0, height=350, anchor='ne')

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.set_table_label("Target", 0, 0, 10, 10)
        self.set_table_label("Position", 0, 1, 10, 10)
        self.set_table_label("Orientation", 0, 2, 10, 10)
        self.set_table_label("execution time (sec)", 0, 3, 10, 10)
        self.set_table_label("Trajectory", 0, 4, 10, 10)
        self.set_table_label("Swap", 0, 5, 10, 10)
        self.set_table_label("Delete?", 0, 6, 10, 10)

    def set_table_label(self, text, row, column, padx, pady, columnspan=1, 
                        font=('Helvatical bold', 10)):
        tk.Label(self.grid_frame, text=text, font=font).grid(row=row, 
                                                             column=column, 
                                                             padx=padx, 
                                                             pady=pady, 
                                                             columnspan=columnspan)

    def add_box(self, position, orientation):
        row_entry = tk.Label(self.grid_frame, text="Target " + str(len(self.targets) + 1), font=('Helvatical bold', 8))
        row_entry.grid(row=len(self.targets) + 1, column=0, padx=10, pady=10, columnspan=1)
        self.targets.append(row_entry)

        row_entry = tk.Entry(self.grid_frame, width=20, font=('Helvatical bold', 8))
        row_entry.grid(row=len(self.positions) + 1, column=1, columnspan=1)
        row_entry.insert(0, position)
        self.positions.append(row_entry)

        row_entry = tk.Entry(self.grid_frame, width=20, font=('Helvatical bold', 8))
        row_entry.grid(row=len(self.orientations) + 1, column=2, columnspan=1)
        row_entry.insert(0, orientation)
        self.orientations.append(row_entry)

        row_entry = tk.Entry(self.grid_frame, width=10, font=('Helvatical bold', 8))
        row_entry.grid(row=len(self.execution_time) + 1, column=3, columnspan=1)
        self.execution_time.append(row_entry)


        row_entry = ttk.Combobox(self.grid_frame, width=10, values=('Joint', 'Cartesian'))
        row_entry.grid(row=len(self.swap) + 1, column=4, columnspan=1)
        row_entry.current(0)
        self.trajectory.append(row_entry)

        row_entry = ttk.Combobox(self.grid_frame, width=10)
        row_entry.grid(row=len(self.swap) + 1, column=5, columnspan=1)
        row_entry.bind("<<ComboboxSelected>>", self.callback_swap)
        self.swap.append(row_entry)
        
        var = tk.IntVar()
        row_entry = tk.Checkbutton(self.grid_frame, variable=var)
        row_entry.grid(row=len(self.delete_target) + 1, column=6, columnspan=1)
        self.delete_target.append([row_entry, var])

        self.update_combobox_values()

    def callback_swap(self, event):
        swap_idx, target_idx = None, None
        for i, widget in enumerate(self.swap):
            s1 = widget.get()
            if len(s1) != 0:
                swap_idx = i
                for j, swap_widget in enumerate(self.targets):
                    if s1 == swap_widget['text']:
                        target_idx = j

        self.swap_row(swap_idx, target_idx)
    
    @staticmethod
    def swap_entry(object_name, s_idx, t_indx):
        tmp = object_name[s_idx].get()
        object_name[s_idx].delete(0, tk.END)
        object_name[s_idx].insert(0, object_name[t_indx].get())
        object_name[t_indx].delete(0, tk.END)
        object_name[t_indx].insert(0, tmp)

    def swap_row(self, swap_index, target_index):
        for idx in range(len(self.targets)):
            self.targets[idx]['text'] = "Target " + str(idx + 1)

        self.swap_entry(self.positions, swap_index, target_index)
        self.swap_entry(self.orientations, swap_index, target_index)
        self.swap_entry(self.execution_time, swap_index, target_index)

        tmp = self.trajectory[swap_index].get()
        self.trajectory[swap_index].set(self.trajectory[target_index].get())
        self.trajectory[target_index].set(tmp)

        tmp = self.trajectory[swap_index]
        self.trajectory[swap_index] = self.trajectory[target_index]
        self.trajectory[target_index] = tmp

        self.swap[swap_index].set('')
        self.swap[target_index].set('')

        self.send_data("update_tcp_plan", [swap_index, target_index])

    def update_combobox_values(self):
        swap_variables = []
        for widget in self.targets:
            swap_variables.append(widget['text'])

        for idx, target_widget in enumerate(self.targets):
            choice_variables = [x for x in swap_variables if x != target_widget['text']]
            self.swap[idx]['values'] = tuple(choice_variables)

    def delete_row(self):
        delete_target_points = []
        for idx, widget in enumerate(self.delete_target):
            _, var = widget
            if var.get():
                delete_target_points.append(idx)

        if delete_target_points:
            self.remove(delete_target_points)

            for idx in range(len(self.targets)):
                self.targets[idx]['text'] = "Target " + str(idx + 1)

            self.update_combobox_values()
            self.send_data("delete_tcp_position", delete_target_points)

    def remove(self, delete_rows):
        for idx in reversed(delete_rows):
            self.targets[idx].grid_remove()
            self.targets.pop(idx)

            self.positions[idx].grid_remove()
            self.positions.pop(idx)

            self.orientations[idx].grid_remove()
            self.orientations.pop(idx)

            self.execution_time[idx].grid_remove()
            self.execution_time.pop(idx)

            self.trajectory[idx].grid_remove()
            self.trajectory.pop(idx)

            self.swap[idx].grid_remove()
            self.swap.pop(idx)

            self.delete_target[idx][0].grid_remove()
            self.delete_target.pop(idx)

    def send_data(self, action, data):
        # put current slider values in queue
        if not self.value_queue.empty():
            _ = self.value_queue.get()
            self.value_queue.put((action, data))
        else:
            self.value_queue.put((action, data))


class RobotControlApp(tk.Frame, DynamicGrid):
    joint_scales = []
    joint_labels = []

    workspace_scales = []
    workspace_labels = []

    def __init__(self, master, input_values, value_queue, simulation_queue):
        super(RobotControlApp, self).__init__()

        self.master = master
        self.workspace = input_values

        self.value_queue = value_queue
        self.simulation_queue = simulation_queue

        self.robot_control = ttk.Notebook(self.master, height=600, width=800)
        self.workspace_tab = tk.Frame(self.robot_control)
        self.target_tab = tk.Frame(self.robot_control)
        DynamicGrid.__init__(self, self.target_tab)

        self.master.geometry("800x700")
        self.master.minsize(800, 700)
        self.master.maxsize(800, 700)

        self.robot_control.add(self.workspace_tab, text='Workspace')
        self.robot_control.add(self.target_tab, text='Targets')
        self.robot_control.grid(row=0, column=0)

        self.error_label = tk.Label(self.target_tab, text="", font=('Helvatical bold', 9), fg='red')
        self.error_label.place(x=15, y=420)

        self.add_position_button = tk.Button(self.workspace_tab,
                                             text="ADD NEW TARGET",
                                             command=self.add_position,
                                             height=5, width=27)
        self.reset_button = tk.Button(self.target_tab,
                                      text="RESET",
                                      command=self.reset,
                                      height=5, width=27)
        self.execute_button = tk.Button(self.target_tab,
                                        text="EXECUTE",
                                        command=self.execute,
                                        height=5, width=27)

        self.delete_target_button = tk.Button(self.target_tab, 
                                              text="DELETE",
                                              command=self.delete_row,
                                              height=5, width=27)

        self.add_position_button.place(x=250, y=475)
        self.reset_button.place(x=15, y=475)
        self.execute_button.place(x=230, y=475)
        self.delete_target_button.place(x=445, y=475)

        self.workspace_control()

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
                                                   length=650,
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
            ttk.Label(self.workspace_tab, text=max_value + unit).place(x=740, y=y)
            self.workspace_labels.append(tk.Label(self.workspace_tab,
                                                  text=start_value + unit))
            self.workspace_labels[i].place(x=400, y=name_y)

            y = y + 60
            name_y = name_y + 60

    def update_slider(self, val):
        slider_values = []
        for idx, slider in enumerate(self.workspace_scales):
            slider_values.append(slider.get())

            if idx >= 3:
                self.workspace_labels[idx]['text'] = str(
                    (round(rad_2_grad(slider.get()), 4))) + "°"
            else:
                self.workspace_labels[idx]['text'] = str(
                    round(slider.get(), 4)) + "m"

        self.send_data("tcp_control", slider_values)

    def read_sliders(self):
        values = []
        for slider in self.workspace_scales:
            values.append(slider.get())

        return values

    def add_position(self):
        slider = self.read_sliders()
        
        self.send_data("add_tcp_position", slider)

        while True:
            if not self.simulation_queue.empty():
                result = self.simulation_queue.get()
                break

        if result[0]:
            position, orientation = [round(x, 4) for x in slider[:3]], [round(x, 4) for x in slider[3:]]
            self.error_label['text'] = ""
            self.add_box(position, orientation)
        else:
            self.error_label['text'] = "Error: Added Position not graspable"

    def reset(self):
        self.send_data("reset", None)

        delete_target_points = []
        for idx in range(len(self.targets)):
            delete_target_points.append(idx)

        if delete_target_points:
            self.remove(delete_target_points)

    def execute(self):
        execution_time = [self.execution_time[i].get() for i in range(len(self.execution_time))]
        execution_pattern = [self.trajectory[i].get() for i in range(len(self.trajectory))]

        # TODO if execution time is empty send nothing and error message
        if not execution_time.count('') > 0:
            self.error_label['text'] = ""

            execution_time = [int(round(float(execution_time[i]))) for i in range(len(execution_time))]

            for idx, value in enumerate(execution_pattern):
                if value == 'Joint':
                    execution_pattern[idx] = 1
                else:
                    execution_pattern[idx] = 0

            self.send_data("execute_trajectory", (execution_time, execution_pattern))
        else:
            self.error_label['text'] = "Error: Set missing execution time"


def run_app(input_values, value_queue, simulation_queue):
    root = tk.Tk(className='robot-planning')
    robot_control_app = RobotControlApp(root,
                                        input_values,
                                        value_queue,
                                        simulation_queue)
    robot_control_app.mainloop()
