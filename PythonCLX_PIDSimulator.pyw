import threading
import time
import random
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint
from pylogix import PLC


class PeriodicInterval(threading.Thread):
    """
    A class for running a task function periodically at a specified interval.
    """

    def __init__(self, task_function, period):
        """
        Initialize the PeriodicInterval thread.
        """
        super().__init__()
        self.daemon = True
        self.task_function = task_function
        self.period = period
        self.i = 0
        self.t0 = time.time()
        self.stop_event = threading.Event()
        self.locker = threading.Lock()
        self.start()

    def sleep(self):
        """
        Sleep for the remaining time to meet the specified period.
        """
        self.i += 1
        delta = self.t0 + self.period * self.i - time.time()
        if delta > 0:
            time.sleep(delta)

    def run(self):
        """
        Start the thread and execute the task_function periodically.
        """
        while not self.stop_event.is_set():
            with self.locker:
                self.task_function()
            self.sleep()

    def stop(self):
        """
        Set the stop event to terminate the periodic task execution.
        """
        self.stop_event.set()


class FOPDTModel(object):
    def __init__(self, CV, ModelData):
        """
        Initialize the FOPDTModel.
        """
        self.CV = CV
        self.Gain, self.TimeConstant, self.DeadTime, self.Bias = ModelData

    def calc(self, PV, ts):
        """
        Calculate the derivative of the process variable.
        """
        if (ts - self.DeadTime) <= 0:
            um = 0
        elif int(ts - self.DeadTime) >= len(self.CV):
            um = self.CV[-1]
        else:
            um = self.CV[int(ts - self.DeadTime)]
        dydt = (-(PV - self.Bias) + self.Gain * um) / (self.TimeConstant)
        return dydt

    def update(self, PV, ts):
        """
        Update the process variable using the model.
        """
        y = odeint(self.calc, PV, ts)
        return y[-1]


class PIDSimForCLX(object):
    def __init__(self):
        self.reset()
        self.gui_setup()
        model = (float(self.model_gain.get()), float(self.model_tc.get()), float(self.model_dt.get()), float(self.model_bias.get()))
        self.process = FOPDTModel(self.CV, model)
        self.comm = PLC()

    def reset(self):
        self.scan_count = 0
        self.PV = np.zeros(0)
        self.CV = np.zeros(0)
        self.SP = np.zeros(0)
        self.looper = None
        self.anim = None

    def gui_setup(self):
        self.root = ctk.CTk()
        self.root.title("PID Simulator for CLX")
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.offset = 7
        self.toolbar = 73

        # Configure GUI window size and appearance
        self.root.resizable(True, True)
        self.root.geometry(f"{int(self.screen_width/2)}x{self.screen_height-self.toolbar}+{-self.offset}+0")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Add a frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(expand=True, fill=ctk.BOTH)

        # Text tags setup
        self.pv_text = ctk.StringVar()
        self.cv_text = ctk.StringVar()
        self.sp_text = ctk.StringVar()
        self.gui_status = ctk.StringVar()
        self.pvtag = ctk.CTkEntry(self.main_frame, width=250)
        self.cvtag = ctk.CTkEntry(self.main_frame, width=250)
        self.sptag = ctk.CTkEntry(self.main_frame, width=250)
        self.ip = ctk.CTkEntry(self.main_frame)
        self.slot = ctk.CTkEntry(self.main_frame, width=60)
        self.model_gain = ctk.CTkEntry(self.main_frame, width=60)
        self.model_tc = ctk.CTkEntry(self.main_frame, width=60)
        self.model_dt = ctk.CTkEntry(self.main_frame, width=60)
        self.model_bias = ctk.CTkEntry(self.main_frame, width=60)
        # Default Values
        self.sptag.insert(0, "SP")
        self.pvtag.insert(0, "PID_PV")
        self.cvtag.insert(0, "PID_CV")
        self.ip.insert(0, "192.168.123.100")
        self.slot.insert(0, "2")
        self.model_gain.insert(0, "1.45")
        self.model_tc.insert(0, "62.3")
        self.model_dt.insert(0, "10.1")
        self.model_bias.insert(0, "13.5")

        # Column 0
        # CTkLabels
        ctk.CTkLabel(self.main_frame, text="Tag").grid(row=0, column=0, padx=10, pady=10, sticky=ctk.NSEW)
        ctk.CTkLabel(self.main_frame, text="SP:").grid(row=1, column=0, padx=10, pady=10, sticky=ctk.NSEW)
        ctk.CTkLabel(self.main_frame, text="PV:").grid(row=2, column=0, padx=10, pady=10, sticky=ctk.NSEW)
        ctk.CTkLabel(self.main_frame, text="CV:").grid(row=3, column=0, padx=10, pady=10, sticky=ctk.NSEW)
        # Row 4 = Button
        # Row 5 = Button
        ctk.CTkLabel(self.main_frame, text="PLC IP Address:").grid(row=6, column=0, padx=10, pady=10, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, text="PLC Slot:").grid(row=7, column=0, padx=10, pady=10, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, text="Model Gain:").grid(row=8, column=0, padx=10, pady=10, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, text="Time Constant:").grid(row=9, column=0, padx=10, pady=10, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, text="Dead Time:").grid(row=10, column=0, padx=10, pady=10, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, text="Model Bias:").grid(row=11, column=0, padx=10, pady=10, sticky=ctk.W)

        # Column 1
        # CTkLabels
        ctk.CTkLabel(self.main_frame, text="Value").grid(row=0, column=1, padx=10, pady=10, sticky=ctk.NSEW)
        ctk.CTkLabel(self.main_frame, textvariable=self.sp_text).grid(row=1, column=1, padx=10, pady=10, sticky=ctk.NSEW)
        ctk.CTkLabel(self.main_frame, textvariable=self.pv_text).grid(row=2, column=1, padx=10, pady=10, sticky=ctk.NSEW)
        ctk.CTkLabel(self.main_frame, textvariable=self.cv_text).grid(row=3, column=1, padx=10, pady=10, sticky=ctk.NSEW)
        # Row 4 = Button
        # Row 5 = Button
        self.ip.grid(row=6, column=1, padx=10, pady=10, sticky=ctk.NSEW)
        self.slot.grid(row=7, column=1, padx=10, pady=10, sticky=ctk.W)
        self.model_gain.grid(row=8, column=1, padx=10, pady=10, sticky=ctk.W)
        self.model_tc.grid(row=9, column=1, padx=10, pady=10, sticky=ctk.W)
        self.model_dt.grid(row=10, column=1, padx=10, pady=10, sticky=ctk.W)
        self.model_bias.grid(row=11, column=1, padx=10, pady=10, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, text="Last Error:").grid(row=12, column=0, padx=10, columnspan=1, pady=10, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, textvariable=self.gui_status, wraplength=375).grid(row=12, column=1, padx=10, columnspan=6, pady=10, sticky=ctk.W)
        ctk.CTkLabel(self.main_frame, text="Seconds").grid(row=9, column=1, padx=30, pady=10, sticky=ctk.E)
        ctk.CTkLabel(self.main_frame, text="Seconds").grid(row=10, column=1, padx=30, pady=10, sticky=ctk.E)

        # Column 2
        # Actual PLC TagName
        ctk.CTkLabel(self.main_frame, text="PLC Tag").grid(row=0, column=2, padx=10, pady=10)
        self.sptag.grid(row=1, column=2, padx=10, pady=10, sticky=ctk.NSEW)
        self.pvtag.grid(row=2, column=2, padx=10, pady=10, sticky=ctk.NSEW)
        self.cvtag.grid(row=3, column=2, padx=10, pady=10, sticky=ctk.NSEW)

        # Buttons
        # Start Button Placement
        self.button_start = ctk.CTkButton(self.main_frame, text="Start Simulator", command=lambda: [self.start()])
        self.button_start.grid(row=4, column=0, columnspan=1, padx=10, pady=10, sticky=ctk.NSEW)

        # Stop Button Placement
        self.button_stop = ctk.CTkButton(self.main_frame, text="Stop Simulator", command=lambda: [self.stop()])
        self.button_stop.grid(row=4, column=1, columnspan=1, padx=10, pady=10, sticky=ctk.NSEW)
        self.button_stop.configure(state=ctk.DISABLED)

        # Trend Button Placement
        self.button_livetrend = ctk.CTkButton(self.main_frame, text="Show Trend", command=lambda: [self.show_live_trend()])
        self.button_livetrend.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky=ctk.NSEW)
        self.button_livetrend.configure(state=ctk.DISABLED)

    def start(self):
        try:
            self.pre_flight_checks()

        except Exception as e:
            self.gui_status.set(str(e))

        else:
            self.looper = PeriodicInterval(self.thread_get_data, 0.1)
            self.live_trend()

    def pre_flight_checks(self):
        self.comm.IPAddress = self.ip.get()
        self.comm.ProcessorSlot = int(self.slot.get())
        self.comm.SocketTimeout = 10.0

        self.read_tag_list = [self.cvtag.get(), self.sptag.get()]
        self.write_tag = self.pvtag.get()

        ret = self.comm.Read([self.cvtag.get(), self.sptag.get(), self.pvtag.get()])
        if any(x.Value is None for x in ret):
            raise Exception("Check PLC and Tag Configuration")
        else:
            self.comm.SocketTimeout = 0.5

        self.reset()
        self.gui_status.set("")
        self.process.Gain = float(self.model_gain.get())
        self.process.TimeConstant = float(self.model_tc.get()) * 10  # Due to sample rate of 0.1 seconds
        self.process.DeadTime = float(self.model_dt.get()) * 10  # Due to sample rate of 0.1 seconds
        self.process.Bias = float(self.model_bias.get())
        # Configure Gui
        self.button_stop.configure(state=ctk.NORMAL)
        self.button_livetrend.configure(state=ctk.DISABLED)
        self.button_start.configure(state=ctk.DISABLED)
        self.ip.configure(state=ctk.DISABLED)
        self.slot.configure(state=ctk.DISABLED)
        self.model_gain.configure(state=ctk.DISABLED)
        self.model_tc.configure(state=ctk.DISABLED)
        self.model_dt.configure(state=ctk.DISABLED)
        self.model_bias.configure(state=ctk.DISABLED)
        self.pvtag.configure(state=ctk.DISABLED)
        self.cvtag.configure(state=ctk.DISABLED)
        self.sptag.configure(state=ctk.DISABLED)

    def thread_get_data(self):
        try:
            ret = self.comm.Read(self.read_tag_list)
            ret_values = [x.Value for x in ret]
            ret_states = [x.Status for x in ret]
            gui_tags = [self.cv_text, self.sp_text]
            for i in range(len(ret_values)):
                if ret_states[i] == "Success":
                    gui_tags[i].set(round(ret_values[i], 3))
                else:
                    self.gui_status.set(ret_states[i])
            if not all(ret_values):
                raise Exception(ret_states[0])
            # Store Data when it is read
            self.CV = np.append(self.CV, ret_values[0])
            self.SP = np.append(self.SP, ret_values[1])
            # Send CV to Process
            self.process.CV = self.CV
            ts = [self.scan_count, self.scan_count + 1]
            # Get new PV value
            if self.PV.size > 1:
                pv = self.process.update(self.PV[-1], ts)
            else:
                pv = self.process.update(float(self.model_bias.get()), ts)
            # Add Noise between -0.1 and 0.1
            noise = (random.randint(0, 10) / 100) - 0.05
            # Store PV
            self.PV = np.append(self.PV, pv[0] + noise)
            # Write PV to PLC
            write = self.comm.Write(self.write_tag, self.PV[-1])
            if write.Status == "Success":
                self.pv_text.set(round(write.Value, 2))
            else:
                self.gui_status.set(write.Status)

        except Exception as e:
            self.gui_status.set("Error: " + str(e))

        else:
            self.scan_count += 1

    def stop(self):
        try:
            self.button_start.configure(state=ctk.NORMAL)
            self.button_livetrend.configure(state=ctk.DISABLED)
            self.button_stop.configure(state=ctk.DISABLED)
            self.ip.configure(state=ctk.NORMAL)
            self.slot.configure(state=ctk.NORMAL)
            self.model_gain.configure(state=ctk.NORMAL)
            self.model_tc.configure(state=ctk.NORMAL)
            self.model_dt.configure(state=ctk.NORMAL)
            self.pvtag.configure(state=ctk.NORMAL)
            self.cvtag.configure(state=ctk.NORMAL)
            self.sptag.configure(state=ctk.NORMAL)
            self.model_bias.configure(state=ctk.NORMAL)
            if self.anim and len(plt.get_fignums()) > 0:
                self.anim.pause()
                self.anim = None
            if self.looper:
                self.looper.stop()
                self.looper = None
            time.sleep(0.1)
            self.comm.Close()
            plt.close("all")

        except Exception as e:
            self.gui_status.set("Stop Error: " + str(e))

    def live_trend(self):
        # Set up the figure
        fig = plt.figure()
        self.ax = plt.axes()
        (SP,) = self.ax.plot([], [], lw=2, color="Red", label="SP")
        (CV,) = self.ax.plot([], [], lw=2, color="Green", label="CV")
        (PV,) = self.ax.plot([], [], lw=2, color="Blue", label="PV")

        # Setup Func
        def init():
            SP.set_data([], [])
            PV.set_data([], [])
            CV.set_data([], [])
            plt.ylabel("EU")
            plt.xlabel("Time (min)")
            plt.suptitle("Live Data")
            plt.legend(loc="upper right")

        # Loop here
        def animate(i):
            try:
                x = np.arange(len(self.SP), dtype=int)
                x = x / 600  # Convert mS to Minutes
                SP.set_data(x, self.SP)
                CV.set_data(x, self.CV)
                PV.set_data(x, self.PV)
                self.ax.relim()
                self.ax.autoscale_view()
            except Exception as e:
                self.gui_status.set("Plot Error: " + str(e))

        # Live Data
        self.anim = animation.FuncAnimation(fig, animate, init_func=init, frames=60, interval=1000)

        mngr = plt.get_current_fig_manager()
        mngr.window.geometry(f"{int(self.screen_width/2)}x{self.screen_height-self.toolbar}+{int(self.screen_width/2)-self.offset+1}+0")
        plt.gcf().canvas.mpl_connect("close_event", self.on_plot_close)
        plt.show()

    def on_plot_close(self, event):
        if self.looper:
            self.button_livetrend.configure(state=ctk.NORMAL)

    def show_live_trend(self):
        self.button_livetrend.configure(state=ctk.DISABLED)
        open_plots = plt.get_fignums()
        if len(open_plots) == 0:
            self.live_trend()


if __name__ == "__main__":
    gui_app = PIDSimForCLX()
    gui_app.root.mainloop()
