#!/usr/bin/env python3
import math
import time
import rclpy
from crazyflie_hover_logger import CrazyflieHover

TOP_URI = "radio://1/90/2M/E7E7E7EA06"
TOP_ODOM_TOPIC = "optitrack/odom"
TOP_HOVER_Z = 1.40
LOWER_HOVER_X = 0.00
LOWER_HOVER_Y = 0.00
DOWNWASH_BIAS_X = 0.12
FREQUENCY_HZ = 0.50
PEAK_SPEED_MPS = 0.50
RAMP_CYCLES = 2
STEADY_CYCLES = 20 # 65s
TAKEOFF_TIME = 3.5
TRANSIT_TIME = 3.0
SETTLE_TIME = 3.0

class CrazyflieSineExcitation(CrazyflieHover):
    def __init__(self):
        super().__init__("crazyflie_sine_excitation", TOP_URI, TOP_ODOM_TOPIC, "crazyflie_sine")
        self.hover_z = TOP_HOVER_Z
        self.omega = 2.0 * math.pi * FREQUENCY_HZ
        self.amplitude = PEAK_SPEED_MPS / self.omega
        self.center_x = LOWER_HOVER_X + DOWNWASH_BIAS_X
        self.center_y = LOWER_HOVER_Y
        self.ramp_time = RAMP_CYCLES / FREQUENCY_HZ
        self.steady_time = STEADY_CYCLES / FREQUENCY_HZ
        self.signal_time = 2.0 * self.ramp_time + self.steady_time
        self.data.update({"signal_frequency_hz": FREQUENCY_HZ, "signal_peak_speed_mps": PEAK_SPEED_MPS, "signal_amplitude_m": self.amplitude, "signal_phase_rad": 0.0, "signal_envelope": 0.0, "signal_input": 0.0, "analysis_valid": 0})
        self.get_logger().info(f"Sine: f={FREQUENCY_HZ:.2f} Hz, A={self.amplitude:.3f} m, vmax={PEAK_SPEED_MPS:.2f} m/s")

    @staticmethod
    def envelope(t, duration, rising):
        s = max(0.0, min(t / duration, 1.0))
        h = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        dh = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
        ddh = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
        return (h, dh, ddh) if rising else (1.0 - h, -dh, -ddh)

    def sine_reference(self, t):
        if t < self.ramp_time:
            gain, gain_dot, gain_ddot = self.envelope(t, self.ramp_time, True)
            phase_name, phase_id = "sine_ramp_up", 4
        elif t < self.ramp_time + self.steady_time:
            gain, gain_dot, gain_ddot = 1.0, 0.0, 0.0
            phase_name, phase_id = "sine_steady", 5
        else:
            gain, gain_dot, gain_ddot = self.envelope(t - self.ramp_time - self.steady_time, self.ramp_time, False)
            phase_name, phase_id = "sine_ramp_down", 6
        phase = self.omega * t
        sine, cosine = math.sin(phase), math.cos(phase)
        x = self.center_x + self.amplitude * gain * sine
        vx = self.amplitude * (gain_dot * sine + gain * self.omega * cosine)
        ax = self.amplitude * (gain_ddot * sine + 2.0 * gain_dot * self.omega * cosine - gain * self.omega**2 * sine)
        return x, vx, ax, phase, gain, gain * sine, int(phase_name == "sine_steady"), phase_name, phase_id

    def hold(self, x, y, z, duration):
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < duration:
            self.send_setpoint(x, y, z)
            time.sleep(1.0 / self.command_rate)

    def move_xy(self, x0, y0, xf, yf, z, duration):
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < duration:
            t = time.perf_counter() - start
            x, vx, ax = self.vertical_reference(x0, xf, t, duration)
            y, vy, ay = self.vertical_reference(y0, yf, t, duration)
            self.send_setpoint(x, y, z, vx=vx, vy=vy, ax=ax, ay=ay)
            time.sleep(1.0 / self.command_rate)
        self.send_setpoint(xf, yf, z)

    def hover_mission(self):
        x0, y0, z0 = self.get_position()
        self.get_logger().info(f"Initial position: x={x0:.2f}, y={y0:.2f}, z={z0:.2f}")
        with self.cf_lock:
            self.cf.commander.send_notify_setpoint_stop()
        self.set_phase("takeoff", 1)
        self.run_vertical_motion(x0, y0, z0, self.hover_z, TAKEOFF_TIME)
        self.set_phase("move_above_lower_drone", 2)
        self.move_xy(x0, y0, self.center_x, self.center_y, self.hover_z, TRANSIT_TIME)
        self.set_phase("settle", 3)
        self.hold(self.center_x, self.center_y, self.hover_z, SETTLE_TIME)
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < self.signal_time:
            t = time.perf_counter() - start
            x, vx, ax, phase, gain, signal, valid, phase_name, phase_id = self.sine_reference(t)
            with self.lock:
                self.data.update({"signal_phase_rad": phase, "signal_envelope": gain, "signal_input": signal, "analysis_valid": valid})
            self.set_phase(phase_name, phase_id)
            self.send_setpoint(x, self.center_y, self.hover_z, vx=vx, ax=ax)
            time.sleep(1.0 / self.command_rate)
        self.set_phase("settle_after_signal", 7)
        self.hold(self.center_x, self.center_y, self.hover_z, SETTLE_TIME)
        self.set_phase("return_to_launch", 8)
        self.move_xy(self.center_x, self.center_y, x0, y0, self.hover_z, TRANSIT_TIME)
        self.set_phase("landing", 9)
        self.run_vertical_motion(x0, y0, self.hover_z, z0, TAKEOFF_TIME)
        self.set_phase("landed", 10)
        self.stop_motors()
        self.get_logger().info(f"Finished. Data saved to {self.csv_path}")
        self.shutdown_flag.set()

def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieSineExcitation()
    try:
        while rclpy.ok() and not node.shutdown_flag.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
