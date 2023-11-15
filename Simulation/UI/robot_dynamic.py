from Simulation.utils.utils import load_numpy_array

from tkinter import ttk
import tkinter as tk
import numpy as np


class RobotControlApp(tk.Frame):
    torque_scales = []
    torque_labels = []

    acceleration_scales = []
    acceleration_labels = []

    mode = 1
    flag = False

    def __init__(self, master, input_values, value_queue, simulation_queue):
        super(RobotControlApp, self).__init__()
        self.master = master
        self.torque_limits = input_values[0]
        self.acceleration_limits = input_values[1]
        self.start_force = input_values[2]
        self.value_queue = value_queue
        self.simulation_queue = simulation_queue

        self.notebook = ttk.Notebook(self.master, height=600, width=450)
        self.acceleration_tab = tk.Frame(self.notebook)
        self.torque_tab = tk.Frame(self.notebook)

        self.master.geometry("450x600")
        self.master.minsize(450, 600)
        self.master.maxsize(450, 600)

        self.notebook.add(self.acceleration_tab, text='Acceleration')
        self.notebook.add(self.torque_tab, text='Torque')
        self.notebook.pack(expand=1, fill="both")

        self.acceleration_control()
        self.torque_control()

        self.reset_button = tk.Button(self.master, text="Reset Simulation",
                                      command=self.reset_simulation, height=5,
                                      width=27)
        
        self.reset_button.place(x=125, y=475)    
        self.update()

    def reset_simulation(self):
        # set mode for update
        self.mode = 'reset'

        reset_qdd = np.zeros(len(self.acceleration_limits))
        reset_tau = load_numpy_array("Simulation/Data/force_solution.npy")

        self.flag = False
        
        self.set_sliders(self.torque_scales, reset_tau)
        self.set_labels(self.torque_labels, reset_tau, " Nm")

        self.set_sliders(self.acceleration_scales, reset_qdd)
        self.set_labels(self.acceleration_labels, reset_tau, " rad/s^2")
        
        slider_values = self.read_sliders()
        self.value_queue.put(["reset", slider_values])

        self.flag = True    

    def acceleration_control(self):
        y = 30
        name_y = 0
        for i, limit in enumerate(self.acceleration_limits):
            title = "Joint " + str(i+1)
            min_label_text = str(round(-limit, 2)) + "rad/s^2"
            max_label_text = str(round(limit, 2)) + "rad/s^2"
            current_position_text = str(round(0.0, 2)) + "rad/s^2"
                
            ttk.Label(self.acceleration_tab, font=('Helvatical bold', 12),  text=title).place(x=20, y=name_y)
            ttk.Label(self.acceleration_tab, text=min_label_text).place(x=20, y=y)
            ttk.Label(self.acceleration_tab, text=max_label_text).place(x=390, y=y)

            self.acceleration_scales.append(ttk.Scale(self.acceleration_tab, 
                                            from_=-limit, 
                                            to=limit,  
                                            length=300, 
                                            value=0,
                                            orient=tk.HORIZONTAL))
            self.acceleration_scales[-1].bind("<ButtonRelease-1>", self.accel_sliders)
            self.acceleration_labels.append(tk.Label(self.acceleration_tab, 
                                            text=current_position_text))

            self.acceleration_scales[i].place(x=80, y=y)
            self.acceleration_labels[i].place(x=225, y=name_y)
            
            y = y + 60
            name_y = name_y + 60

    def torque_control(self):
        y = 30
        name_y = 0
        for i, limit in enumerate(self.torque_limits):
            title = "Joint " + str(i+1)
            min_label_text = str(round(-limit, 2)) + "Nm"
            max_label_text = str(round(limit, 2)) + "Nm"
            current_position_text = str(round(self.start_force[i], 2)) + "Nm"
                
            ttk.Label(self.torque_tab, font=('Helvatical bold', 12),  text=title).place(x=20, y=name_y)
            ttk.Label(self.torque_tab, text=min_label_text).place(x=20, y=y)
            ttk.Label(self.torque_tab, text=max_label_text).place(x=390, y=y)

            self.torque_scales.append(ttk.Scale(self.torque_tab, 
                                                from_=-limit,
                                                to=limit,
                                                length=300,
                                                value=self.start_force[i],
                                                orient=tk.HORIZONTAL))
            self.torque_scales[-1].bind("<ButtonRelease-1>", self.torque_sliders)
            self.torque_labels.append(tk.Label(self.torque_tab, 
                                               text=current_position_text))
            
            self.torque_scales[i].place(x=80, y=y)
            self.torque_labels[i].place(x=225, y=name_y)

            y = y + 60
            name_y = name_y + 60

    def torque_sliders(self, val):
        # set mode for update app
        self.mode = "torque"

        slider_values = self.read_sliders()
        self.update_slider(slider_values, "torque")

    def accel_sliders(self, val):
        # set mode for update app
        self.mode = 'acceleration'
        
        slider_values = self.read_sliders()
        self.update_slider(slider_values, 'acceleration')

    def update_slider(self, sliders, mode):
        i = 0
        # update labels in app if slider gets activated
        for idx, value in enumerate(sliders):           
            if idx < len(self.torque_limits):
                self.torque_labels[idx]['text'] = str(round(value, 2)) + " Nm"
            else:
                self.acceleration_labels[i]['text'] = str(round(value, 2)) + " rad/s^2"
            
            if idx >= len(self.torque_limits):
                i = i + 1
        
        if self.flag:
            # Note callback will automatically called if scale is set. 
            # use flag for checking if its a user call
            # put current slider values in queue
            if not self.value_queue.empty():
                _ = self.value_queue.get() 
                self.value_queue.put([mode, sliders])
            else:
                self.value_queue.put([mode, sliders])

    def read_sliders(self):
        values = []
        for slider in self.torque_scales:
            values.append(slider.get())

        for slider in self.acceleration_scales:
            values.append(slider.get())

        return values
    
    @staticmethod
    def set_sliders(scales, values):
        for slider, value in zip(scales, values):
            slider.set(value)

    @staticmethod
    def set_labels(labels, values, unit):
        for label, value in zip(labels, values):
            label['text'] = str(round(value, 2)) + unit 
    
    def update(self):
        if not self.simulation_queue.empty():
            self.flag = False

            new_tau, new_qdd = None, None
            while not self.simulation_queue.empty():
                new_tau, new_qdd = self.simulation_queue.get()
            
            if self.mode == 'reset':
                self.set_sliders(self.torque_scales, new_tau)
                self.set_labels(self.torque_labels, new_tau, " Nm")
                self.set_sliders(self.acceleration_scales, new_qdd)
                self.set_labels(self.acceleration_labels, new_qdd, " rad/s^2")
            if self.mode == 'acceleration':
                self.set_sliders(self.torque_scales, new_tau)
                self.set_labels(self.torque_labels, new_tau, " Nm")
            if self.mode == "torque":
                self.set_sliders(self.acceleration_scales, new_qdd)
                self.set_labels(self.acceleration_labels, new_qdd, " rad/s^2")

            self.flag = True
        
        self.master.after(250, self.update)


def run_app(input_values, value_queue, simulation_queue):
    root = tk.Tk(className='robot-control')
    robot_control_app = RobotControlApp(root, 
                                        input_values, 
                                        value_queue, 
                                        simulation_queue)   
    robot_control_app.mainloop()
