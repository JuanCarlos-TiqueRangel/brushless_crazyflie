#!/usr/bin/env python3
"""
crazyflie_contraction_tube_figure8.py

Disturbance-aware contraction/tube wrapper for a brushless Crazyflie.

What this is:
    High-level ROS2 + cflib controller that keeps the onboard Crazyflie
    Mellinger controller as the fast inner-loop controller, but replaces the
    plain reference streaming with a disturbance-aware full-state reference.

Core idea:
    1) Generate a smooth full-state figure-8 reference: p_ref, v_ref, a_ref.
    2) Estimate an unknown acceleration disturbance from OptiTrack velocity:
           d_hat ~= a_meas - a_cmd_previous
    3) Add a contraction/tube correction to the commanded acceleration:
           a_cmd = a_ref - Kp e_p - Kd e_v - Ktube sat(e_p / rho) - d_hat
       where e_p = p - p_ref and e_v = v - v_ref.
    4) Send p_ref, v_ref, a_cmd to the Crazyflie full-state commander.

Why this is useful:
    Plain full-state Mellinger sends nominal position/velocity/acceleration.
    This wrapper makes the acceleration feedforward adaptive to persistent
    wind/bias and more aggressive when the drone leaves a tracking tube.

Important safety notes:
    - This is a high-level wrapper, not a replacement for the firmware-level
      attitude/rate controller.
    - Start with a small trajectory: radius_x <= 0.5, radius_y <= 0.25,
      hover_z around 0.5-0.7 m, and low acceleration limits.
    - Keep an emergency stop ready.

Run example:
    ros2 run YOUR_PACKAGE crazyflie_contraction_tube_figure8.py \
        --ros-args \
        -p uri:=radio://0/80/2M/E7E7E7E705 \
        -p odom_topic:=optitrack/odom \
        -p radius_x:=0.5 \
        -p radius_y:=0.25 \
        -p loop_duration_s:=10.0
"""

import csv
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.reset_estimator import reset_estimator


Vec3 = Tuple[float, float, float]


@dataclass
class ControllerChoice:
    # Crazyflie runtime IDs commonly used in the firmware:
    #   1 = PID
    #   2 = Mellinger
    #   3 = INDI
    #   4 = Brescianini
    #   5 = Lee
    # For full-state setpoints, Mellinger is the safest default.
    controller_id: str = "2"
    controller_name: str = "MellingerPlusContractionTubeDOB"


class CrazyflieContractionTubeFigure8(Node):
    def __init__(self) -> None:
        super().__init__("crazyflie_contraction_tube_figure8")

        # -----------------------------
        # Basic communication / mission parameters
        # -----------------------------
        self.declare_parameter("uri", "radio://0/80/2M/E7E7E7E705")
        self.declare_parameter("odom_topic", "optitrack/odom")

        self.declare_parameter("extpos_rate_hz", 100.0)
        self.declare_parameter("cmd_rate_hz", 100.0)
        self.declare_parameter("log_rate_hz", 50.0)
        self.declare_parameter("odom_timeout_s", 0.20)
        self.declare_parameter("estimator_reset_samples", 50)

        self.declare_parameter("auto_start", True)
        self.declare_parameter("arm_on_start", True)

        self.declare_parameter("use_current_position_as_center", True)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)

        self.declare_parameter("hover_z", 0.8)
        self.declare_parameter("yaw_rad", 0.0)

        self.declare_parameter("takeoff_duration_s", 2.5)
        self.declare_parameter("pre_hold_s", 1.0)
        self.declare_parameter("loop_duration_s", 8.0)
        self.declare_parameter("num_loops", 4)
        self.declare_parameter("radius_x", 2.0)
        self.declare_parameter("radius_y", 1.0)
        self.declare_parameter("ramp_in_s", 2.0)
        self.declare_parameter("ramp_out_s", 2.0)
        self.declare_parameter("post_hold_s", 0.5)
        self.declare_parameter("landing_duration_s", 2.5)

        # -----------------------------
        # Disturbance-aware contraction/tube parameters
        # -----------------------------
        self.declare_parameter("enable_tube_feedback", True)
        self.declare_parameter("enable_disturbance_observer", True)

        # Contraction-like second-order tracking gains.
        # Kp = omega^2, Kd = 2*zeta*omega.
        self.declare_parameter("omega_xy", 3.0)
        self.declare_parameter("zeta_xy", 1.05)
        self.declare_parameter("omega_z", 3.5)
        self.declare_parameter("zeta_z", 1.10)

        # Tube term: extra bounded acceleration when error leaves the tube.
        self.declare_parameter("tube_radius_xy", 0.12)
        self.declare_parameter("tube_radius_z", 0.08)
        self.declare_parameter("tube_gain_xy", 0.45)
        self.declare_parameter("tube_gain_z", 0.35)

        # Disturbance observer and acceleration saturation.
        self.declare_parameter("disturbance_filter_tau_s", 0.35)
        self.declare_parameter("disturbance_deadband", 0.04)
        self.declare_parameter("max_disturbance_xy", 1.2)
        self.declare_parameter("max_disturbance_z", 1.0)
        self.declare_parameter("max_xy_acc_cmd", 2.0)
        self.declare_parameter("max_z_acc_cmd", 1.5)

        # Safety monitor.
        self.declare_parameter("max_tracking_error_m", 0.75)
        self.declare_parameter("max_speed_mps", 10.0)
        self.declare_parameter("abort_on_large_error", True)

        self.declare_parameter("enable_csv_logging", True)

        # -----------------------------
        # Read parameters
        # -----------------------------
        self.uri = str(self.get_parameter("uri").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.extpos_rate_hz = float(self.get_parameter("extpos_rate_hz").value)
        self.cmd_rate_hz = float(self.get_parameter("cmd_rate_hz").value)
        self.log_rate_hz = float(self.get_parameter("log_rate_hz").value)
        self.odom_timeout_s = float(self.get_parameter("odom_timeout_s").value)
        self.estimator_reset_samples = int(self.get_parameter("estimator_reset_samples").value)

        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.arm_on_start = bool(self.get_parameter("arm_on_start").value)

        self.use_current_position_as_center = bool(
            self.get_parameter("use_current_position_as_center").value
        )
        self.center_x_param = float(self.get_parameter("center_x").value)
        self.center_y_param = float(self.get_parameter("center_y").value)

        self.hover_z = float(self.get_parameter("hover_z").value)
        self.yaw_rad = float(self.get_parameter("yaw_rad").value)

        self.takeoff_duration_s = float(self.get_parameter("takeoff_duration_s").value)
        self.pre_hold_s = float(self.get_parameter("pre_hold_s").value)
        self.loop_duration_s = float(self.get_parameter("loop_duration_s").value)
        self.num_loops = int(self.get_parameter("num_loops").value)
        self.radius_x = float(self.get_parameter("radius_x").value)
        self.radius_y = float(self.get_parameter("radius_y").value)
        self.ramp_in_s = float(self.get_parameter("ramp_in_s").value)
        self.ramp_out_s = float(self.get_parameter("ramp_out_s").value)
        self.post_hold_s = float(self.get_parameter("post_hold_s").value)
        self.landing_duration_s = float(self.get_parameter("landing_duration_s").value)

        self.enable_tube_feedback = bool(self.get_parameter("enable_tube_feedback").value)
        self.enable_disturbance_observer = bool(
            self.get_parameter("enable_disturbance_observer").value
        )

        self.omega_xy = float(self.get_parameter("omega_xy").value)
        self.zeta_xy = float(self.get_parameter("zeta_xy").value)
        self.omega_z = float(self.get_parameter("omega_z").value)
        self.zeta_z = float(self.get_parameter("zeta_z").value)

        self.tube_radius_xy = float(self.get_parameter("tube_radius_xy").value)
        self.tube_radius_z = float(self.get_parameter("tube_radius_z").value)
        self.tube_gain_xy = float(self.get_parameter("tube_gain_xy").value)
        self.tube_gain_z = float(self.get_parameter("tube_gain_z").value)

        self.disturbance_filter_tau_s = float(
            self.get_parameter("disturbance_filter_tau_s").value
        )
        self.disturbance_deadband = float(self.get_parameter("disturbance_deadband").value)
        self.max_disturbance_xy = float(self.get_parameter("max_disturbance_xy").value)
        self.max_disturbance_z = float(self.get_parameter("max_disturbance_z").value)
        self.max_xy_acc_cmd = float(self.get_parameter("max_xy_acc_cmd").value)
        self.max_z_acc_cmd = float(self.get_parameter("max_z_acc_cmd").value)

        self.max_tracking_error_m = float(self.get_parameter("max_tracking_error_m").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.abort_on_large_error = bool(self.get_parameter("abort_on_large_error").value)

        self.enable_csv_logging = bool(self.get_parameter("enable_csv_logging").value)

        # -----------------------------
        # Shared measured state
        # -----------------------------
        self.state_lock = threading.Lock()
        self.cf_lock = threading.Lock()

        self.odom_received = False
        self.odom_count = 0
        self.last_odom_wall_time = 0.0

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.connected = False
        self.estimator_ready = False
        self.estimator_reset_started = False
        self.mission_started = False
        self.mission_abort = threading.Event()
        self.shutdown_flag = threading.Event()

        self.scf: Optional[SyncCrazyflie] = None
        self.cf: Optional[Crazyflie] = None

        # -----------------------------
        # Reference / command / observer state for logging
        # -----------------------------
        self.x_ref = 0.0
        self.y_ref = 0.0
        self.z_ref = 0.0
        self.vx_ref = 0.0
        self.vy_ref = 0.0
        self.vz_ref = 0.0
        self.ax_ref = 0.0
        self.ay_ref = 0.0
        self.az_ref = 0.0

        self.ax_cmd = 0.0
        self.ay_cmd = 0.0
        self.az_cmd = 0.0

        self.ex = 0.0
        self.ey = 0.0
        self.ez = 0.0
        self.evx = 0.0
        self.evy = 0.0
        self.evz = 0.0
        self.tube_energy = 0.0

        self.dhat_x = 0.0
        self.dhat_y = 0.0
        self.dhat_z = 0.0

        self._observer_initialized = False
        self._last_control_wall_time = 0.0
        self._last_v_meas: Vec3 = (0.0, 0.0, 0.0)
        self._last_acc_cmd: Vec3 = (0.0, 0.0, 0.0)

        # -----------------------------
        # ROS subscriber
        # -----------------------------
        self.sub_odom = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        # -----------------------------
        # CSV logging
        # -----------------------------
        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / "logs"
        self.csv_file = None
        self.csv_writer = None

        if self.enable_csv_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.csv_path = str(
                self.log_dir / f"contraction_tube.csv"
            )
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                [
                    "t_wall",
                    "x_ref", "y_ref", "z_ref",
                    "vx_ref", "vy_ref", "vz_ref",
                    "ax_ref", "ay_ref", "az_ref",
                    "ax_cmd", "ay_cmd", "az_cmd",
                    "x_meas", "y_meas", "z_meas",
                    "vx_meas", "vy_meas", "vz_meas",
                    "ex", "ey", "ez", "evx", "evy", "evz",
                    "dhat_x", "dhat_y", "dhat_z",
                    "tube_energy",
                ]
            )
            self.log_timer = self.create_timer(1.0 / self.log_rate_hz, self.log_timer_callback)
            self.get_logger().info(f"CSV logging to {self.csv_path}")

        # -----------------------------
        # Crazyflie connect + timers
        # -----------------------------
        self.connect_to_crazyflie()

        self.extpos_timer = self.create_timer(
            1.0 / self.extpos_rate_hz,
            self.extpos_timer_callback,
        )

        self.get_logger().info("Crazyflie contraction/tube figure-8 node started.")

    # --------------------------------------------------
    # Utility vector functions
    # --------------------------------------------------
    @staticmethod
    def norm2(x: float, y: float) -> float:
        return math.sqrt(x * x + y * y)

    @staticmethod
    def norm3(v: Vec3) -> float:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    @staticmethod
    def clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(value, hi))

    @staticmethod
    def deadband(value: float, band: float) -> float:
        if abs(value) < band:
            return 0.0
        return value

    @staticmethod
    def limit_xy_z(v: Vec3, max_xy: float, max_z: float) -> Vec3:
        x, y, z = v
        xy = math.sqrt(x * x + y * y)
        if xy > max_xy > 1e-9:
            scale = max_xy / xy
            x *= scale
            y *= scale
        z = max(-max_z, min(z, max_z))
        return x, y, z

    @staticmethod
    def sat_unit(value: float) -> float:
        return max(-1.0, min(value, 1.0))

    @staticmethod
    def yaw_to_quaternion(yaw_rad: float):
        half = 0.5 * yaw_rad
        return [0.0, 0.0, math.sin(half), math.cos(half)]  # [qx, qy, qz, qw]

    # --------------------------------------------------
    # Crazyflie setup
    # --------------------------------------------------
    def connect_to_crazyflie(self) -> None:
        self.get_logger().info(f"Connecting to Crazyflie at {self.uri} ...")
        cflib.crtp.init_drivers()

        self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache="./cache"))
        self.scf.open_link()
        self.cf = self.scf.cf
        self.connected = True

        self.get_logger().info("Connected to Crazyflie.")

        try:
            self.cf.param.set_value("stabilizer.estimator", "2")
            time.sleep(0.2)
            self.get_logger().info("Set stabilizer.estimator = 2 (Kalman).")
        except Exception as exc:
            self.get_logger().warn(f"Could not set Kalman estimator: {exc}")

        try:
            self.cf.param.set_value("stabilizer.controller", ControllerChoice.controller_id)
            time.sleep(0.2)
            self.get_logger().info(
                f"Set stabilizer.controller = {ControllerChoice.controller_id} "
                f"({ControllerChoice.controller_name})"
            )
        except Exception as exc:
            self.get_logger().warn(f"Could not set controller: {exc}")

        if self.arm_on_start:
            try:
                self.cf.platform.send_arming_request(True)
                time.sleep(1.0)
                self.get_logger().info("Arming request sent.")
            except Exception as exc:
                self.get_logger().warn(f"Automatic arming failed: {exc}")

    # --------------------------------------------------
    # ROS callbacks / state
    # --------------------------------------------------
    def odom_callback(self, msg: Odometry) -> None:
        with self.state_lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            self.z = msg.pose.pose.position.z

            self.vx = msg.twist.twist.linear.x
            self.vy = msg.twist.twist.linear.y
            self.vz = msg.twist.twist.linear.z

            self.odom_received = True
            self.odom_count += 1
            self.last_odom_wall_time = time.time()

    def get_state_copy(self) -> Tuple[float, float, float, float, float, float]:
        with self.state_lock:
            return self.x, self.y, self.z, self.vx, self.vy, self.vz

    def odom_is_fresh(self) -> bool:
        if not self.odom_received:
            return False
        return (time.time() - self.last_odom_wall_time) < self.odom_timeout_s

    # --------------------------------------------------
    # CSV logging
    # --------------------------------------------------
    def log_timer_callback(self) -> None:
        if self.csv_writer is None or not self.odom_received:
            return

        x, y, z, vx, vy, vz = self.get_state_copy()
        self.csv_writer.writerow(
            [
                time.time(),
                self.x_ref, self.y_ref, self.z_ref,
                self.vx_ref, self.vy_ref, self.vz_ref,
                self.ax_ref, self.ay_ref, self.az_ref,
                self.ax_cmd, self.ay_cmd, self.az_cmd,
                x, y, z,
                vx, vy, vz,
                self.ex, self.ey, self.ez,
                self.evx, self.evy, self.evz,
                self.dhat_x, self.dhat_y, self.dhat_z,
                self.tube_energy,
            ]
        )
        if self.csv_file is not None:
            self.csv_file.flush()

    # --------------------------------------------------
    # External position feed
    # --------------------------------------------------
    def extpos_timer_callback(self) -> None:
        if not self.connected or self.cf is None or not self.odom_received:
            return

        x, y, z, _, _, _ = self.get_state_copy()

        try:
            with self.cf_lock:
                self.cf.extpos.send_extpos(x, y, z)
        except Exception as exc:
            self.get_logger().error(f"Failed to send extpos: {exc}")
            return

        if (
            not self.estimator_reset_started
            and not self.estimator_ready
            and self.odom_count >= self.estimator_reset_samples
        ):
            self.estimator_reset_started = True
            threading.Thread(target=self.reset_estimator_worker, daemon=True).start()

    def reset_estimator_worker(self) -> None:
        if self.scf is None:
            return

        self.get_logger().info("Resetting estimator...")
        try:
            reset_estimator(self.scf)
            self.estimator_ready = True
            self.get_logger().info("Estimator is ready.")

            if self.auto_start and not self.mission_started:
                self.mission_started = True
                threading.Thread(target=self.mission_worker, daemon=True).start()

        except Exception as exc:
            self.get_logger().error(f"Estimator reset failed: {exc}")
            self.estimator_reset_started = False

    # --------------------------------------------------
    # Trajectory helpers
    # --------------------------------------------------
    @staticmethod
    def smoothstep5(t: float, T: float) -> Tuple[float, float, float]:
        """
        Quintic smooth step from 0 to 1 over [0, T].
        Returns value, first derivative, second derivative.
        """
        if T <= 1e-9:
            return 1.0, 0.0, 0.0

        s = max(0.0, min(t / T, 1.0))

        val = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        dval_ds = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
        d2val_ds2 = 60.0 * s - 180.0 * s**2 + 120.0 * s**3

        dval_dt = dval_ds / T
        d2val_dt2 = d2val_ds2 / (T * T)
        return val, dval_dt, d2val_dt2

    def hover_state(self, x: float, y: float, z: float):
        return (x, y, z), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    def vertical_quintic(
        self,
        x: float,
        y: float,
        z0: float,
        z1: float,
        t: float,
        T: float,
    ):
        s, ds, dds = self.smoothstep5(t, T)
        z = z0 + (z1 - z0) * s
        vz = (z1 - z0) * ds
        az = (z1 - z0) * dds
        return (x, y, z), (0.0, 0.0, vz), (0.0, 0.0, az)

    def envelope(self, t: float, T_total: float) -> Tuple[float, float, float]:
        """
        Smooth amplitude envelope:
        - ramp from 0 -> 1 during ramp_in_s
        - hold near 1
        - ramp from 1 -> 0 during ramp_out_s
        """
        Tin = min(self.ramp_in_s, 0.45 * T_total)
        Tout = min(self.ramp_out_s, 0.45 * T_total)

        if t < Tin:
            return self.smoothstep5(t, Tin)

        if t > T_total - Tout:
            val, dval, ddval = self.smoothstep5(t - (T_total - Tout), Tout)
            return 1.0 - val, -dval, -ddval

        return 1.0, 0.0, 0.0

    def figure8_state(
        self,
        cx: float,
        cy: float,
        zf: float,
        t: float,
        T_total: float,
    ):
        """
        Smooth figure-8 with analytic position, velocity, acceleration.

        Base curve:
            dx = rx * sin(theta)
            dy = 0.5 * ry * sin(2 theta)
        with a start/stop envelope.
        """
        omega = 2.0 * math.pi / self.loop_duration_s
        theta = omega * t

        dx = self.radius_x * math.sin(theta)
        dy = 0.5 * self.radius_y * math.sin(2.0 * theta)

        ddx_dt = self.radius_x * omega * math.cos(theta)
        ddy_dt = self.radius_y * omega * math.cos(2.0 * theta)

        d2dx_dt2 = -self.radius_x * omega * omega * math.sin(theta)
        d2dy_dt2 = -2.0 * self.radius_y * omega * omega * math.sin(2.0 * theta)

        a, ad, add = self.envelope(t, T_total)

        x = cx + a * dx
        y = cy + a * dy
        z = zf

        vx = ad * dx + a * ddx_dt
        vy = ad * dy + a * ddy_dt
        vz = 0.0

        ax = add * dx + 2.0 * ad * ddx_dt + a * d2dx_dt2
        ay = add * dy + 2.0 * ad * ddy_dt + a * d2dy_dt2
        az = 0.0

        return (x, y, z), (vx, vy, vz), (ax, ay, az)

    # --------------------------------------------------
    # Disturbance-aware contraction/tube controller
    # --------------------------------------------------
    def reset_disturbance_observer(self) -> None:
        self._observer_initialized = False
        self._last_control_wall_time = 0.0
        self._last_v_meas = (0.0, 0.0, 0.0)
        self._last_acc_cmd = (0.0, 0.0, 0.0)
        self.dhat_x = 0.0
        self.dhat_y = 0.0
        self.dhat_z = 0.0

    def update_disturbance_observer(self, v_meas: Vec3, now: float) -> Vec3:
        """
        Low-pass estimate of unknown acceleration disturbance.

        Approximate translational dynamics at the high-level interface:
            v_dot = a_cmd + d
        so:
            d ~= measured_acceleration - previous_commanded_acceleration

        This is intentionally conservative and saturated because OptiTrack
        velocity can be noisy.
        """
        if not self.enable_disturbance_observer:
            return (0.0, 0.0, 0.0)

        if not self._observer_initialized:
            self._observer_initialized = True
            self._last_control_wall_time = now
            self._last_v_meas = v_meas
            self._last_acc_cmd = (self.ax_cmd, self.ay_cmd, self.az_cmd)
            return (self.dhat_x, self.dhat_y, self.dhat_z)

        dt = now - self._last_control_wall_time
        if dt < 1e-4 or dt > 0.20:
            self._last_control_wall_time = now
            self._last_v_meas = v_meas
            self._last_acc_cmd = (self.ax_cmd, self.ay_cmd, self.az_cmd)
            return (self.dhat_x, self.dhat_y, self.dhat_z)

        a_meas = (
            (v_meas[0] - self._last_v_meas[0]) / dt,
            (v_meas[1] - self._last_v_meas[1]) / dt,
            (v_meas[2] - self._last_v_meas[2]) / dt,
        )

        raw = (
            a_meas[0] - self._last_acc_cmd[0],
            a_meas[1] - self._last_acc_cmd[1],
            a_meas[2] - self._last_acc_cmd[2],
        )

        raw = (
            self.deadband(raw[0], self.disturbance_deadband),
            self.deadband(raw[1], self.disturbance_deadband),
            self.deadband(raw[2], self.disturbance_deadband),
        )
        raw = self.limit_xy_z(raw, self.max_disturbance_xy, self.max_disturbance_z)

        tau = max(self.disturbance_filter_tau_s, 1e-3)
        alpha = math.exp(-dt / tau)
        self.dhat_x = alpha * self.dhat_x + (1.0 - alpha) * raw[0]
        self.dhat_y = alpha * self.dhat_y + (1.0 - alpha) * raw[1]
        self.dhat_z = alpha * self.dhat_z + (1.0 - alpha) * raw[2]

        # Limit the filtered estimate as well.
        self.dhat_x, self.dhat_y, self.dhat_z = self.limit_xy_z(
            (self.dhat_x, self.dhat_y, self.dhat_z),
            self.max_disturbance_xy,
            self.max_disturbance_z,
        )

        self._last_control_wall_time = now
        self._last_v_meas = v_meas
        self._last_acc_cmd = (self.ax_cmd, self.ay_cmd, self.az_cmd)

        return (self.dhat_x, self.dhat_y, self.dhat_z)

    def contraction_tube_acceleration(
        self,
        pos_ref: Vec3,
        vel_ref: Vec3,
        acc_ref: Vec3,
    ) -> Vec3:
        """
        Compute corrected acceleration command for the full-state setpoint.

        The contraction-like part is a second-order stable error dynamics:
            e_ddot + Kd e_dot + Kp e = residual_disturbance

        The tube term adds bounded robust acceleration once error approaches
        the tracking tube radius. This is the practical version of tightening
        around the reference trajectory.
        """
        now = time.time()
        x, y, z, vx, vy, vz = self.get_state_copy()
        pos = (x, y, z)
        vel = (vx, vy, vz)

        dhat = self.update_disturbance_observer(vel, now)

        ex = pos[0] - pos_ref[0]
        ey = pos[1] - pos_ref[1]
        ez = pos[2] - pos_ref[2]
        evx = vel[0] - vel_ref[0]
        evy = vel[1] - vel_ref[1]
        evz = vel[2] - vel_ref[2]

        kp_xy = self.omega_xy * self.omega_xy
        kd_xy = 2.0 * self.zeta_xy * self.omega_xy
        kp_z = self.omega_z * self.omega_z
        kd_z = 2.0 * self.zeta_z * self.omega_z

        a_fb = (
            -kp_xy * ex - kd_xy * evx,
            -kp_xy * ey - kd_xy * evy,
            -kp_z * ez - kd_z * evz,
        )

        if self.enable_tube_feedback:
            tube = (
                -self.tube_gain_xy * self.sat_unit(ex / max(self.tube_radius_xy, 1e-3)),
                -self.tube_gain_xy * self.sat_unit(ey / max(self.tube_radius_xy, 1e-3)),
                -self.tube_gain_z * self.sat_unit(ez / max(self.tube_radius_z, 1e-3)),
            )
        else:
            tube = (0.0, 0.0, 0.0)

        # Disturbance compensation is negative feedback of the estimated bias.
        a_cmd = (
            acc_ref[0] + a_fb[0] + tube[0] - dhat[0],
            acc_ref[1] + a_fb[1] + tube[1] - dhat[1],
            acc_ref[2] + a_fb[2] + tube[2] - dhat[2],
        )
        a_cmd = self.limit_xy_z(a_cmd, self.max_xy_acc_cmd, self.max_z_acc_cmd)

        # Store for logging.
        self.x_ref, self.y_ref, self.z_ref = pos_ref
        self.vx_ref, self.vy_ref, self.vz_ref = vel_ref
        self.ax_ref, self.ay_ref, self.az_ref = acc_ref
        self.ax_cmd, self.ay_cmd, self.az_cmd = a_cmd
        self.ex, self.ey, self.ez = ex, ey, ez
        self.evx, self.evy, self.evz = evx, evy, evz
        self.tube_energy = (
            (ex / max(self.tube_radius_xy, 1e-3)) ** 2
            + (ey / max(self.tube_radius_xy, 1e-3)) ** 2
            + (ez / max(self.tube_radius_z, 1e-3)) ** 2
        )

        return a_cmd

    def safety_check_or_raise(self) -> None:
        err = self.norm3((self.ex, self.ey, self.ez))
        speed = self.norm3((self.vx, self.vy, self.vz))

        if err > self.max_tracking_error_m:
            msg = f"large tracking error: {err:.3f} m > {self.max_tracking_error_m:.3f} m"
            if self.abort_on_large_error:
                raise RuntimeError(msg)
            self.get_logger().warn(msg)

        if speed > self.max_speed_mps:
            msg = f"large measured speed: {speed:.3f} m/s > {self.max_speed_mps:.3f} m/s"
            if self.abort_on_large_error:
                raise RuntimeError(msg)
            self.get_logger().warn(msg)

    def stream_disturbance_aware_full_state(
        self,
        pos_ref: Vec3,
        vel_ref: Vec3,
        acc_ref: Vec3,
        yaw_rad: float,
        rollrate_deg_s: float = 0.0,
        pitchrate_deg_s: float = 0.0,
        yawrate_deg_s: float = 0.0,
    ) -> None:
        if self.cf is None:
            return

        acc_cmd = self.contraction_tube_acceleration(pos_ref, vel_ref, acc_ref)
        quat = self.yaw_to_quaternion(yaw_rad)

        with self.cf_lock:
            self.cf.commander.send_full_state_setpoint(
                list(pos_ref),
                list(vel_ref),
                list(acc_cmd),
                quat,
                rollrate_deg_s,
                pitchrate_deg_s,
                yawrate_deg_s,
            )

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(remaining)

    # --------------------------------------------------
    # Mission
    # --------------------------------------------------
    def mission_worker(self) -> None:
        if self.cf is None:
            return

        dt = 1.0 / self.cmd_rate_hz

        try:
            self.get_logger().info(
                f"Starting figure-8 with {ControllerChoice.controller_name}: "
                f"omega_xy={self.omega_xy:.2f}, omega_z={self.omega_z:.2f}, "
                f"tube_xy={self.tube_radius_xy:.2f}, tube_z={self.tube_radius_z:.2f}"
            )

            if not self.odom_is_fresh():
                raise RuntimeError("OptiTrack odometry is stale before mission start.")

            x0, y0, z0, _, _, _ = self.get_state_copy()

            if self.use_current_position_as_center:
                cx = x0
                cy = y0
            else:
                cx = self.center_x_param
                cy = self.center_y_param

            zf = self.hover_z
            self.reset_disturbance_observer()

            # If previous HLC command was active, release it.
            with self.cf_lock:
                self.cf.commander.send_notify_setpoint_stop()

            # -----------------------------
            # Takeoff
            # -----------------------------
            self.get_logger().info(f"Takeoff to z={zf:.2f} m")
            start = time.time()
            k = 0
            while not self.shutdown_flag.is_set():
                t = time.time() - start
                if t >= self.takeoff_duration_s:
                    break
                if not self.odom_is_fresh():
                    raise RuntimeError("OptiTrack odometry became stale during takeoff.")

                pos, vel, acc = self.vertical_quintic(cx, cy, z0, zf, t, self.takeoff_duration_s)
                self.stream_disturbance_aware_full_state(pos, vel, acc, self.yaw_rad)
                self.safety_check_or_raise()

                k += 1
                self.sleep_until(start + k * dt)

            # -----------------------------
            # Hover before trajectory
            # -----------------------------
            self.get_logger().info("Holding hover before trajectory...")
            start = time.time()
            k = 0
            while not self.shutdown_flag.is_set():
                t = time.time() - start
                if t >= self.pre_hold_s:
                    break
                if not self.odom_is_fresh():
                    raise RuntimeError("OptiTrack odometry became stale during pre-hold.")

                pos, vel, acc = self.hover_state(cx, cy, zf)
                self.stream_disturbance_aware_full_state(pos, vel, acc, self.yaw_rad)
                self.safety_check_or_raise()

                k += 1
                self.sleep_until(start + k * dt)

            # -----------------------------
            # Figure-8
            # -----------------------------
            T_fig = self.loop_duration_s * self.num_loops
            self.get_logger().info(
                f"Running figure-8: loops={self.num_loops}, T={T_fig:.2f}s, "
                f"rx={self.radius_x:.2f}, ry={self.radius_y:.2f}"
            )

            start = time.time()
            k = 0
            while not self.shutdown_flag.is_set():
                t = time.time() - start
                if t >= T_fig:
                    break
                if not self.odom_is_fresh():
                    raise RuntimeError("OptiTrack odometry became stale during trajectory.")

                pos, vel, acc = self.figure8_state(cx, cy, zf, t, T_fig)
                self.stream_disturbance_aware_full_state(pos, vel, acc, self.yaw_rad)
                self.safety_check_or_raise()

                k += 1
                self.sleep_until(start + k * dt)

            # -----------------------------
            # Hover after trajectory
            # -----------------------------
            self.get_logger().info("Holding hover after trajectory...")
            start = time.time()
            k = 0
            while not self.shutdown_flag.is_set():
                t = time.time() - start
                if t >= self.post_hold_s:
                    break
                if not self.odom_is_fresh():
                    raise RuntimeError("OptiTrack odometry became stale during post-hold.")

                pos, vel, acc = self.hover_state(cx, cy, zf)
                self.stream_disturbance_aware_full_state(pos, vel, acc, self.yaw_rad)
                self.safety_check_or_raise()

                k += 1
                self.sleep_until(start + k * dt)

            # -----------------------------
            # Landing
            # -----------------------------
            self.get_logger().info("Landing...")
            start = time.time()
            k = 0
            while not self.shutdown_flag.is_set():
                t = time.time() - start
                if t >= self.landing_duration_s:
                    break
                if not self.odom_is_fresh():
                    raise RuntimeError("OptiTrack odometry became stale during landing.")

                pos, vel, acc = self.vertical_quintic(cx, cy, zf, 0.0, t, self.landing_duration_s)
                self.stream_disturbance_aware_full_state(pos, vel, acc, self.yaw_rad)

                k += 1
                self.sleep_until(start + k * dt)

            self.get_logger().info("Mission completed, stopping motors.")
            self.hard_kill()

        except Exception as exc:
            self.get_logger().error(f"Mission error: {exc}")
            self.abort_and_land()

    # --------------------------------------------------
    # Stop / shutdown
    # --------------------------------------------------
    def abort_and_land(self) -> None:
        if self.cf is None:
            return

        try:
            # Try a short controlled descent using full-state setpoints.
            x, y, z, _, _, _ = self.get_state_copy()
            self.get_logger().warn("Abort requested, attempting controlled landing...")
            start = time.time()
            dt = 1.0 / max(self.cmd_rate_hz, 20.0)
            k = 0
            T = max(1.5, self.landing_duration_s)

            while not self.shutdown_flag.is_set():
                t = time.time() - start
                if t >= T:
                    break
                pos, vel, acc = self.vertical_quintic(x, y, z, 0.0, t, T)
                # During abort landing, use nominal acceleration only and reset observer
                # to avoid noisy disturbance estimates near the ground.
                self.ax_cmd, self.ay_cmd, self.az_cmd = acc
                quat = self.yaw_to_quaternion(self.yaw_rad)
                with self.cf_lock:
                    self.cf.commander.send_full_state_setpoint(
                        list(pos), list(vel), list(acc), quat, 0.0, 0.0, 0.0
                    )
                k += 1
                self.sleep_until(start + k * dt)

        except Exception as exc:
            self.get_logger().warn(f"Controlled landing failed: {exc}")
        finally:
            self.hard_kill()

    def hard_kill(self) -> None:
        if self.cf is None:
            return

        try:
            with self.cf_lock:
                self.cf.commander.send_stop_setpoint()
        except Exception as exc:
            self.get_logger().warn(f"Stop setpoint failed: {exc}")

        try:
            with self.cf_lock:
                self.cf.platform.send_arming_request(False)
        except Exception:
            pass

    def destroy_node(self):
        self.get_logger().warn("Shutting down node, stopping motors...")
        self.shutdown_flag.set()
        self.mission_abort.set()

        try:
            self.hard_kill()
        except Exception:
            pass

        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception as exc:
                self.get_logger().warn(f"Failed to close CSV file: {exc}")
            finally:
                self.csv_file = None
                self.csv_writer = None

        if self.scf is not None:
            try:
                self.scf.close_link()
            except Exception as exc:
                self.get_logger().warn(f"close_link failed: {exc}")

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CrazyflieContractionTubeFigure8()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
