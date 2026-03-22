#!/usr/bin/env python3
import math
import threading
import time
import csv
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


@dataclass
class ControllerChoice:
    # Crazyflie runtime IDs:
    # 2 = Mellinger
    # 4 = Brescianini
    # 5 = Lee
    controller_id: str = "4"
    controller_name: str = "Brescianini"


class CrazyflieFullStateFigure8(Node):
    def __init__(self) -> None:
        super().__init__("crazyflie_fullstate_figure8")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("uri", "radio://0/80/2M/E7E7E7E705")
        self.declare_parameter("odom_topic", "optitrack/odom")

        self.declare_parameter("extpos_rate_hz", 100.0)
        self.declare_parameter("cmd_rate_hz", 100.0)
        self.declare_parameter("log_rate_hz", 20.0)
        self.declare_parameter("odom_timeout_s", 0.20)
        self.declare_parameter("estimator_reset_samples", 50)

        self.declare_parameter("auto_start", True)
        self.declare_parameter("arm_on_start", True)

        self.declare_parameter("use_current_position_as_center", True)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)

        self.declare_parameter("hover_z", 0.6)
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

        self.declare_parameter("enable_csv_logging", True)

        self.uri = self.get_parameter("uri").value
        self.odom_topic = self.get_parameter("odom_topic").value

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

        self.enable_csv_logging = bool(self.get_parameter("enable_csv_logging").value)

        # -----------------------------
        # Shared state
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

        # Current commanded reference for logging
        self.x_ref = 0.0
        self.y_ref = 0.0
        self.z_ref = 0.0
        self.vx_ref = 0.0
        self.vy_ref = 0.0
        self.vz_ref = 0.0
        self.ax_ref = 0.0
        self.ay_ref = 0.0
        self.az_ref = 0.0

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
        # Logging
        # -----------------------------
        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / "logs"
        self.csv_file = None
        self.csv_writer = None

        if self.enable_csv_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.csv_path = str(
                self.log_dir
                / f"figure8_fullstate_{ControllerChoice.controller_name}_{stamp}.csv"
            )
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                [
                    "t_wall",
                    "x_ref", "y_ref", "z_ref",
                    "vx_ref", "vy_ref", "vz_ref",
                    "ax_ref", "ay_ref", "az_ref",
                    "x_meas", "y_meas", "z_meas",
                    "vx_meas", "vy_meas", "vz_meas",
                ]
            )
            self.log_timer = self.create_timer(1.0 / self.log_rate_hz, self.log_timer_callback)
            self.get_logger().info(f"CSV logging to {self.csv_path}")

        # -----------------------------
        # Connect
        # -----------------------------
        self.connect_to_crazyflie()

        # Keep external position streaming
        self.extpos_timer = self.create_timer(
            1.0 / self.extpos_rate_hz,
            self.extpos_timer_callback,
        )

        self.get_logger().info("Crazyflie Full-State Figure-8 node started.")

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

    # def odom_is_fresh(self) -> bool:
    #     if not self.odom_received:
    #         return False
    #     return (time.time() - self.last_odom_wall_time) < self.odom_timeout_s

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
                x, y, z,
                vx, vy, vz,
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
    # Math helpers
    # --------------------------------------------------
    @staticmethod
    def clamp_time_fraction(t: float, T: float) -> float:
        if T <= 0.0:
            return 1.0
        return max(0.0, min(t / T, 1.0))

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

    @staticmethod
    def yaw_to_quaternion(yaw_rad: float):
        half = 0.5 * yaw_rad
        return [0.0, 0.0, math.sin(half), math.cos(half)]  # [qx, qy, qz, qw]

    def set_ref(
        self,
        pos: Tuple[float, float, float],
        vel: Tuple[float, float, float],
        acc: Tuple[float, float, float],
    ) -> None:
        self.x_ref, self.y_ref, self.z_ref = pos
        self.vx_ref, self.vy_ref, self.vz_ref = vel
        self.ax_ref, self.ay_ref, self.az_ref = acc

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(remaining)

    def stream_full_state(
        self,
        pos: Tuple[float, float, float],
        vel: Tuple[float, float, float],
        acc: Tuple[float, float, float],
        yaw_rad: float,
        rollrate_deg_s: float = 0.0,
        pitchrate_deg_s: float = 0.0,
        yawrate_deg_s: float = 0.0,
    ) -> None:
        if self.cf is None:
            return

        self.set_ref(pos, vel, acc)
        quat = self.yaw_to_quaternion(yaw_rad)

        with self.cf_lock:
            self.cf.commander.send_full_state_setpoint(
                list(pos),
                list(vel),
                list(acc),
                quat,
                rollrate_deg_s,
                pitchrate_deg_s,
                yawrate_deg_s,
            )

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
        wrapped in a start/stop envelope so we begin and end at hover.
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
    # Mission
    # --------------------------------------------------
    def mission_worker(self) -> None:
        if self.cf is None:
            return

        dt = 1.0 / self.cmd_rate_hz

        try:
            self.get_logger().info(
                f"Starting full-state figure-8 with {ControllerChoice.controller_name}..."
            )

            # if not self.odom_is_fresh():
            #     raise RuntimeError("OptiTrack odometry is stale before mission start.")

            x0, y0, z0, _, _, _ = self.get_state_copy()

            if self.use_current_position_as_center:
                cx = x0
                cy = y0
            else:
                cx = self.center_x_param
                cy = self.center_y_param

            zf = self.hover_z

            # If some previous HLC command was active, release it
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
                # if not self.odom_is_fresh():
                #     raise RuntimeError("OptiTrack odometry became stale during takeoff.")

                pos, vel, acc = self.vertical_quintic(cx, cy, z0, zf, t, self.takeoff_duration_s)
                self.stream_full_state(pos, vel, acc, self.yaw_rad)

                k += 1
                self.sleep_until(start + k * dt)

            # small hold
            self.get_logger().info("Holding hover before trajectory...")
            start = time.time()
            k = 0
            while not self.shutdown_flag.is_set():
                t = time.time() - start
                if t >= self.pre_hold_s:
                    break
                # if not self.odom_is_fresh():
                #     raise RuntimeError("OptiTrack odometry became stale during pre-hold.")

                pos, vel, acc = self.hover_state(cx, cy, zf)
                self.stream_full_state(pos, vel, acc, self.yaw_rad)

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
                # if not self.odom_is_fresh():
                #     raise RuntimeError("OptiTrack odometry became stale during trajectory.")

                pos, vel, acc = self.figure8_state(cx, cy, zf, t, T_fig)
                self.stream_full_state(pos, vel, acc, self.yaw_rad)

                k += 1
                self.sleep_until(start + k * dt)

            # post hold
            self.get_logger().info("Holding hover after trajectory...")
            start = time.time()
            k = 0
            while not self.shutdown_flag.is_set():
                t = time.time() - start
                if t >= self.post_hold_s:
                    break
                # if not self.odom_is_fresh():
                #     raise RuntimeError("OptiTrack odometry became stale during post-hold.")

                pos, vel, acc = self.hover_state(cx, cy, zf)
                self.stream_full_state(pos, vel, acc, self.yaw_rad)

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
                # if not self.odom_is_fresh():
                #     raise RuntimeError("OptiTrack odometry became stale during landing.")

                pos, vel, acc = self.vertical_quintic(cx, cy, zf, 0.0, t, self.landing_duration_s)
                self.stream_full_state(pos, vel, acc, self.yaw_rad)

                k += 1
                self.sleep_until(start + k * dt)

            # Final motor stop
            self.get_logger().info("Mission completed, stopping motors.")
            self.hard_kill()

        except Exception as exc:
            self.get_logger().error(f"Mission error: {exc}")
            self.hard_kill()

    # --------------------------------------------------
    # Stop / shutdown
    # --------------------------------------------------
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
        node = CrazyflieFullStateFigure8()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()