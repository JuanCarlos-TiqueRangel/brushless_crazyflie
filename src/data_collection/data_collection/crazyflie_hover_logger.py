#!/usr/bin/env python3
import csv
import math
import threading
import time
from datetime import datetime

import cflib.crtp
import rclpy
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.reset_estimator import reset_estimator
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


DEFAULT_NODE_NAME = "crazyflie_hover"
DEFAULT_URI = "radio://0/80/2M/E7E7E7EA03"
DEFAULT_ODOM_TOPIC = "optitrack/odom2"
DEFAULT_CSV_PREFIX = "crazyflie_hover"

LOG_BLOCKS = {
    "imu": (10, [
        "acc.x",
        # "acc.y",
        # "acc.z",
        # # "gyro.x",
        # # "gyro.y",
        # # "gyro.z",
    ]),
}

EXPERIMENT_FIELDS = [
    "host_monotonic_ns",
    "host_unix_ns",
    "ros_timestamp_ns",
    "phase",
    "phase_id",
    "ref_x",
    "ref_y",
    "ref_z",
    "ref_vx",
    "ref_vy",
    "ref_vz",
    "ref_ax",
    "ref_ay",
    "ref_az",
    "opti_x",
    "opti_y",
    "opti_z",
    "opti_vx",
    "opti_vy",
    "opti_vz",
    "opti_qx",
    "opti_qy",
    "opti_qz",
    "opti_qw",
    "opti_roll_deg",
    "opti_pitch_deg",
    "opti_yaw_deg",
    "opti_wx",
    "opti_wy",
    "opti_wz",
    "signal_frequency_hz",
    "signal_peak_speed_mps",
    "signal_amplitude_m",
    "signal_phase_rad",
    "signal_envelope",
    "signal_input",
    "analysis_valid",
]


class CrazyflieHover(Node):
    def __init__(
        self,
        node_name=DEFAULT_NODE_NAME,
        uri=DEFAULT_URI,
        odom_topic=DEFAULT_ODOM_TOPIC,
        csv_prefix=DEFAULT_CSV_PREFIX,
    ):
        super().__init__(node_name)

        self.uri = uri
        self.odom_topic = odom_topic

        self.hover_z = 0.60
        # self.hover_points = [
        #     (0.0, 0.0, 0.6, 10.0),
        #     (0.3, 0.0, 0.6, 10.0),
        #     (0.3, 0.0, 0.9, 10.0),
        #     (0.3, 0.0, 1.2, 10.0),
        #     (0.6, 0.0, 1.2, 10.0),
        #     (0.6, 0.0, 0.9, 10.0),
        #     (0.6, 0.0, 0.6, 10.0),
        #     (0.9, 0.0, 0.6, 10.0),
        #     (0.9, 0.0, 0.9, 10.0),
        #     (0.9, 0.0, 1.2, 10.0),
        # ]

        self.hover_points = [
            (0.6, 0.0, 0.9, 10.0),
            (0.5, 0.0, 0.9, 10.0),
            (0.4, 0.0, 0.9, 10.0),
            (0.3, 0.0, 0.9, 10.0),
            (0.2, 0.0, 0.9, 10.0),
            (0.1, 0.0, 0.9, 10.0),
            (0.0, 0.0, 0.9, 10.0),
            (-0.1, 0.0, 0.9, 10.0),
            (-0.2, 0.0, 0.9, 10.0),
            (-0.3, 0.0, 0.9, 10.0),
            (-0.4, 0.0, 0.9, 10.0),
            (-0.5, 0.0, 0.9, 10.0),
            (-0.6, 0.0, 0.9, 10.0),
        ]

        self.landing_x = 0.90
        self.landing_y = 0.00

        self.takeoff_time = 2.5
        self.move_time = 2.0
        self.landing_time = 5.0
        self.landing_settle_time = 0.5
        self.takeoff_settle_time = 10.0

        self.extpos_rate = 50.0
        self.command_rate = 50.0

        self.lock = threading.Lock()
        self.cf_lock = threading.Lock()
        self.shutdown_flag = threading.Event()

        self.data = {
            "ref_x": math.nan,
            "ref_y": math.nan,
            "ref_z": math.nan,
            "ref_vx": 0.0,
            "ref_vy": 0.0,
            "ref_vz": 0.0,
            "ref_ax": 0.0,
            "ref_ay": 0.0,
            "ref_az": 0.0,
            "opti_x": 0.0,
            "opti_y": 0.0,
            "opti_z": 0.0,
            "opti_vx": math.nan,
            "opti_vy": math.nan,
            "opti_vz": math.nan,
            "opti_qx": math.nan,
            "opti_qy": math.nan,
            "opti_qz": math.nan,
            "opti_qw": math.nan,
            "opti_roll_deg": math.nan,
            "opti_pitch_deg": math.nan,
            "opti_yaw_deg": math.nan,
            "opti_wx": math.nan,
            "opti_wy": math.nan,
            "opti_wz": math.nan,
            "signal_frequency_hz": 0.0,
            "signal_peak_speed_mps": 0.0,
            "signal_amplitude_m": 0.0,
            "signal_phase_rad": 0.0,
            "signal_envelope": 0.0,
            "signal_input": 0.0,
            "analysis_valid": 0,
        }

        self.phase = "waiting"
        self.phase_id = 0
        self.odom_received = False
        self.extpos_sent = 0
        self.estimator_reset_started = False
        self.mission_started = False
        self.link_lost = False

        self.log_configs = []
        self.cf_files = {}
        self.cf_writers = {}
        self.cf_rows = {"imu": 0}
        self.experiment_rows = 0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_path = f"{csv_prefix}_experiment_{stamp}.csv"
        self.imu_path = f"{csv_prefix}_imu_{stamp}.csv"

        self.experiment_file = open(self.experiment_path, "w", newline="")
        self.experiment_writer = csv.DictWriter(
            self.experiment_file,
            fieldnames=EXPERIMENT_FIELDS,
        )
        self.experiment_writer.writeheader()

        self.cf_files["imu"] = open(self.imu_path, "w", newline="")
        self.cf_writers["imu"] = csv.DictWriter(
            self.cf_files["imu"],
            fieldnames=[
                "cf_timestamp_ms",
                "host_monotonic_ns",
                "host_unix_ns",
                *LOG_BLOCKS["imu"][1],
            ],
        )
        self.cf_writers["imu"].writeheader()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos,
        )

        cflib.crtp.init_drivers()

        cf = Crazyflie(rw_cache="./cache")
        cf.connection_lost.add_callback(self.connection_lost)
        self.scf = SyncCrazyflie(self.uri, cf=cf)

        try:
            self.get_logger().info(f"Connecting to {self.uri}...")
            self.scf.open_link()
            self.cf = self.scf.cf

            self.cf.param.set_value("stabilizer.estimator", "2")
            time.sleep(0.2)
            self.cf.param.set_value("stabilizer.controller", "2")
            time.sleep(0.2)

            self.start_cf_logging()

            self.cf.platform.send_arming_request(True)
            time.sleep(1.0)

        except Exception:
            self.close_files()
            try:
                self.scf.close_link()
            except Exception:
                pass
            raise

        self.create_timer(1.0 / self.extpos_rate, self.send_external_position)

        self.get_logger().info("Connected. Logging IMU at 100 Hz.")
        self.get_logger().info(f"IMU: {self.imu_path}")
        self.get_logger().info(f"OptiTrack/reference: {self.experiment_path}")
        self.get_logger().info("Waiting for OptiTrack...")

    def connection_lost(self, link_uri, message):
        self.link_lost = True
        self.shutdown_flag.set()
        self.get_logger().error(f"Connection lost to {link_uri}: {message}")

    def start_cf_logging(self):
        for block_name, (period_ms, variables) in LOG_BLOCKS.items():
            config = LogConfig(name=block_name, period_in_ms=period_ms)

            for variable in variables:
                group, name = variable.split(".", 1)
                if group not in self.cf.log.toc.toc or name not in self.cf.log.toc.toc[group]:
                    raise RuntimeError(f"Crazyflie log variable not available: {variable}")
                config.add_variable(variable)

            self.cf.log.add_config(config)
            config.data_received_cb.add_callback(self.cf_log_callback)
            config.error_cb.add_callback(self.cf_log_error)
            config.start()
            self.log_configs.append(config)

    def cf_log_callback(self, timestamp, data, logconf):
        block_name = logconf.name
        writer = self.cf_writers.get(block_name)
        if writer is None:
            return

        row = {
            "cf_timestamp_ms": int(timestamp),
            "host_monotonic_ns": time.monotonic_ns(),
            "host_unix_ns": time.time_ns(),
        }
        row.update(data)
        writer.writerow(row)

        self.cf_rows[block_name] += 1
        if self.cf_rows[block_name] % 100 == 0:
            self.cf_files[block_name].flush()

    def cf_log_error(self, logconf, message):
        self.get_logger().error(f"Crazyflie log error [{logconf.name}]: {message}")

    @staticmethod
    def quaternion_to_euler(qx, qy, qz, qw):
        roll = math.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        pitch = math.asin(
            max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
        )
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        roll, pitch, yaw = self.quaternion_to_euler(q.x, q.y, q.z, q.w)

        host_monotonic_ns = time.monotonic_ns()
        host_unix_ns = time.time_ns()
        ros_timestamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

        with self.lock:
            self.data.update({
                "opti_x": p.x,
                "opti_y": p.y,
                "opti_z": p.z,
                "opti_vx": v.x,
                "opti_vy": v.y,
                "opti_vz": v.z,
                "opti_qx": q.x,
                "opti_qy": q.y,
                "opti_qz": q.z,
                "opti_qw": q.w,
                "opti_roll_deg": roll,
                "opti_pitch_deg": pitch,
                "opti_yaw_deg": yaw,
                "opti_wx": w.x,
                "opti_wy": w.y,
                "opti_wz": w.z,
            })
            self.odom_received = True

            row = {
                "host_monotonic_ns": host_monotonic_ns,
                "host_unix_ns": host_unix_ns,
                "ros_timestamp_ns": ros_timestamp_ns,
                "phase": self.phase,
                "phase_id": self.phase_id,
            }

            for field in EXPERIMENT_FIELDS:
                if field not in row:
                    row[field] = self.data.get(field, "")

        self.experiment_writer.writerow(row)
        self.experiment_rows += 1
        if self.experiment_rows % 100 == 0:
            self.experiment_file.flush()

    def get_position(self):
        with self.lock:
            return self.data["opti_x"], self.data["opti_y"], self.data["opti_z"]

    def send_external_position(self):
        with self.lock:
            if not self.odom_received or self.shutdown_flag.is_set():
                return
            x = self.data["opti_x"]
            y = self.data["opti_y"]
            z = self.data["opti_z"]

        with self.cf_lock:
            self.cf.extpos.send_extpos(x, y, z)

        self.extpos_sent += 1
        if self.extpos_sent >= 25 and not self.estimator_reset_started:
            self.estimator_reset_started = True
            threading.Thread(target=self.initialize_estimator, daemon=True).start()

    def initialize_estimator(self):
        if self.shutdown_flag.is_set():
            return

        self.get_logger().info("Resetting Kalman estimator...")
        try:
            reset_estimator(self.scf)
        except Exception as exc:
            self.get_logger().error(f"Estimator reset failed: {exc}")
            self.shutdown_flag.set()
            return

        self.get_logger().info("Kalman estimator ready.")
        if not self.mission_started and not self.shutdown_flag.is_set():
            self.mission_started = True
            threading.Thread(target=self.hover_mission, daemon=True).start()

    def set_phase(self, name, number):
        with self.lock:
            self.phase = name
            self.phase_id = number

    def send_setpoint(
        self,
        x,
        y,
        z,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        ax=0.0,
        ay=0.0,
        az=0.0,
    ):
        with self.lock:
            self.data.update({
                "ref_x": x,
                "ref_y": y,
                "ref_z": z,
                "ref_vx": vx,
                "ref_vy": vy,
                "ref_vz": vz,
                "ref_ax": ax,
                "ref_ay": ay,
                "ref_az": az,
            })

        with self.cf_lock:
            self.cf.commander.send_full_state_setpoint(
                [x, y, z],
                [vx, vy, vz],
                [ax, ay, az],
                [0.0, 0.0, 0.0, 1.0],
                0.0,
                0.0,
                0.0,
            )

    @staticmethod
    def smooth_reference(p0, pf, t, duration):
        s = max(0.0, min(t / duration, 1.0))
        p = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        dp = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
        ddp = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
        return p0 + (pf - p0) * p, (pf - p0) * dp, (pf - p0) * ddp

    def run_vertical_motion(self, x, y, z0, zf, duration):
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < duration:
            t = time.perf_counter() - start
            z, vz, az = self.smooth_reference(z0, zf, t, duration)
            self.send_setpoint(x, y, z, vz=vz, az=az)
            time.sleep(1.0 / self.command_rate)

        if not self.shutdown_flag.is_set():
            self.send_setpoint(x, y, zf)

    def run_position_motion(self, x0, y0, z0, xf, yf, zf, duration):
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < duration:
            t = time.perf_counter() - start
            x, vx, ax = self.smooth_reference(x0, xf, t, duration)
            y, vy, ay = self.smooth_reference(y0, yf, t, duration)
            z, vz, az = self.smooth_reference(z0, zf, t, duration)
            self.send_setpoint(
                x,
                y,
                z,
                vx=vx,
                vy=vy,
                vz=vz,
                ax=ax,
                ay=ay,
                az=az,
            )
            time.sleep(1.0 / self.command_rate)

        if not self.shutdown_flag.is_set():
            self.send_setpoint(xf, yf, zf)

    def hover_mission(self):
        x0, y0, z0 = self.get_position()
        self.get_logger().info(f"Initial position: x={x0:.2f}, y={y0:.2f}, z={z0:.2f}")

        with self.cf_lock:
            self.cf.commander.send_notify_setpoint_stop()

        self.set_phase("takeoff", 1)
        self.get_logger().info(f"Taking off to {self.hover_z:.2f} m...")
        self.run_vertical_motion(x0, y0, z0, self.hover_z, self.takeoff_time)

        self.set_phase("takeoff_settle", 2)
        self.get_logger().info(f"Stabilizing for {self.takeoff_settle_time:.1f} seconds...")

        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < self.takeoff_settle_time:
            self.send_setpoint(x0, y0, self.hover_z)
            time.sleep(1.0 / self.command_rate)

        current_x = x0
        current_y = y0
        current_z = self.hover_z
        phase_id = 3

        for point_id, (target_x, target_y, target_z, hover_time) in enumerate(self.hover_points, start=1):
            if self.shutdown_flag.is_set():
                return

            self.set_phase(f"move_to_point_{point_id}", phase_id)
            self.get_logger().info(
                f"Moving to point {point_id}: x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}"
            )

            self.run_position_motion(
                current_x,
                current_y,
                current_z,
                target_x,
                target_y,
                target_z,
                self.move_time,
            )

            phase_id += 1
            self.set_phase(f"hover_point_{point_id}", phase_id)
            self.get_logger().info(f"Hovering at point {point_id} for {hover_time:.1f} seconds...")

            start = time.perf_counter()
            while not self.shutdown_flag.is_set() and time.perf_counter() - start < hover_time:
                self.send_setpoint(target_x, target_y, target_z)
                time.sleep(1.0 / self.command_rate)

            current_x = target_x
            current_y = target_y
            current_z = target_z
            phase_id += 1

        if self.shutdown_flag.is_set():
            return

        self.set_phase("return_to_landing", phase_id)
        self.run_position_motion(
            current_x,
            current_y,
            current_z,
            self.landing_x,
            self.landing_y,
            self.hover_z,
            self.move_time,
        )

        phase_id += 1
        self.set_phase("landing", phase_id)
        self.get_logger().info("Landing...")

        self.run_vertical_motion(self.landing_x, self.landing_y, self.hover_z, z0, self.landing_time)

        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < self.landing_settle_time:
            self.send_setpoint(self.landing_x, self.landing_y, z0)
            time.sleep(1.0 / self.command_rate)

        if self.shutdown_flag.is_set():
            return

        self.set_phase("landed", phase_id + 1)
        self.stop_motors()
        self.get_logger().info("Landed.")
        self.shutdown_flag.set()

    def stop_motors(self):
        with self.cf_lock:
            self.cf.commander.send_stop_setpoint()
            self.cf.platform.send_arming_request(False)

    def close_files(self):
        if hasattr(self, "experiment_file") and not self.experiment_file.closed:
            self.experiment_file.flush()
            self.experiment_file.close()

        for file in self.cf_files.values():
            if not file.closed:
                file.flush()
                file.close()

    def destroy_node(self):
        self.shutdown_flag.set()

        if not self.link_lost:
            try:
                self.stop_motors()
            except Exception:
                pass

        for config in self.log_configs:
            try:
                config.stop()
            except Exception:
                pass

        self.close_files()

        try:
            self.scf.close_link()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieHover()

    try:
        while rclpy.ok() and not node.shutdown_flag.is_set():
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()