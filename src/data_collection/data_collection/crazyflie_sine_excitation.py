#!/usr/bin/env python3
import math
import time
import rclpy
from crazyflie_hover_logger import CrazyflieHover

# TOP_URI = "radio://1/90/2M/E7E7E7EA06" Big drone
TOP_URI = "radio://1/80/2M/E7E7E7E705"
TOP_ODOM_TOPIC = "optitrack/odom"
TOP_HOVER_Z = 1.60
LOWER_HOVER_X = 0.00
LOWER_HOVER_Y = 0.00
DOWNWASH_BIAS_X = 0.12
FREQUENCY_STEPS = [(1.0, 20.0), (0.8, 20.0), (0.6, 20.0)]
PEAK_SPEED_MPS = 0.2
RAMP_CYCLES = 2
TAKEOFF_TIME = 3.5
TRANSIT_TIME = 3.0
SETTLE_TIME = 40.0

class CrazyflieSineExcitation(CrazyflieHover):
    def __init__(self):
        super().__init__("crazyflie_sine_excitation", TOP_URI, TOP_ODOM_TOPIC, "crazyflie_sine")
        self.hover_z = TOP_HOVER_Z
        self.center_x = LOWER_HOVER_X + DOWNWASH_BIAS_X
        self.center_y = LOWER_HOVER_Y
        self.data.update({"signal_frequency_hz": 0.0, "signal_peak_speed_mps": PEAK_SPEED_MPS, "signal_amplitude_m": 0.0, "signal_phase_rad": 0.0, "signal_envelope": 0.0, "signal_input": 0.0, "analysis_valid": 0})
        self.get_logger().info(f"Frequency steps: {FREQUENCY_STEPS}, vmax={PEAK_SPEED_MPS:.2f} m/s")

    @staticmethod
    def envelope(t, duration, rising):
        s = max(0.0, min(t / duration, 1.0))
        h = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        dh = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
        ddh = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
        return (h, dh, ddh) if rising else (1.0 - h, -dh, -ddh)

    def sine_reference(self, t, frequency, phase0, gain, gain_dot, gain_ddot):
        omega = 2.0 * math.pi * frequency
        amplitude = PEAK_SPEED_MPS / omega
        phase = phase0 + omega * t
        sine, cosine = math.sin(phase), math.cos(phase)
        x = self.center_x + amplitude * gain * sine
        vx = amplitude * (gain_dot * sine + gain * omega * cosine)
        ax = amplitude * (gain_ddot * sine + 2.0 * gain_dot * omega * cosine - gain * omega**2 * sine)
        return x, vx, ax, phase, amplitude, gain * sine

    def run_sine(self, frequency, duration, phase0, mode, phase_name, phase_id):
        self.set_phase(phase_name, phase_id)
        self.get_logger().info(f"{phase_name}: {frequency:.1f} Hz for {duration:.1f} s")
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < duration:
            t = time.perf_counter() - start
            if mode == "up":
                gain, gain_dot, gain_ddot = self.envelope(t, duration, True)
            elif mode == "down":
                gain, gain_dot, gain_ddot = self.envelope(t, duration, False)
            else:
                gain, gain_dot, gain_ddot = 1.0, 0.0, 0.0
            x, vx, ax, phase, amplitude, signal = self.sine_reference(t, frequency, phase0, gain, gain_dot, gain_ddot)
            with self.lock:
                self.data.update({"signal_frequency_hz": frequency, "signal_amplitude_m": amplitude, "signal_phase_rad": phase, "signal_envelope": gain, "signal_input": signal, "analysis_valid": int(mode == "steady")})
            self.send_setpoint(x, self.center_y, self.hover_z, vx=vx, ax=ax)
            time.sleep(1.0 / self.command_rate)
        gain, gain_dot, gain_ddot = ((0.0, 0.0, 0.0) if mode == "down" else (1.0, 0.0, 0.0))
        x, vx, ax, phase, amplitude, signal = self.sine_reference(duration, frequency, phase0, gain, gain_dot, gain_ddot)
        with self.lock:
            self.data.update({"signal_frequency_hz": frequency, "signal_amplitude_m": amplitude, "signal_phase_rad": phase, "signal_envelope": gain, "signal_input": signal, "analysis_valid": 0})
        self.send_setpoint(x, self.center_y, self.hover_z, vx=vx, ax=ax)
        return phase

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
        phase = self.run_sine(FREQUENCY_STEPS[0][0], RAMP_CYCLES / FREQUENCY_STEPS[0][0], 0.0, "up", "sine_ramp_up", 4)
        for phase_id, (frequency, duration) in enumerate(FREQUENCY_STEPS, start=5):
            phase = self.run_sine(frequency, duration, phase, "steady", f"sine_{frequency:g}hz", phase_id)
        phase = self.run_sine(FREQUENCY_STEPS[-1][0], RAMP_CYCLES / FREQUENCY_STEPS[-1][0], phase, "down", "sine_ramp_down", 8)
        self.set_phase("settle_after_signal", 9)
        self.hold(self.center_x, self.center_y, self.hover_z, SETTLE_TIME)
        self.set_phase("return_to_launch", 10)
        self.move_xy(self.center_x, self.center_y, x0, y0, self.hover_z, TRANSIT_TIME)
        self.set_phase("landing", 11)
        self.run_vertical_motion(x0, y0, self.hover_z, z0, TAKEOFF_TIME)
        self.set_phase("landed", 12)
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