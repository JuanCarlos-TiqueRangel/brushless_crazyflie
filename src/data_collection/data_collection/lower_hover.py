#!/usr/bin/env python3
import csv
import math
import threading
import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.reset_estimator import reset_estimator

LOG_BLOCKS = {
    "estimate": (40, ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z", "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"]),
    "state_acc": (40, ["stateEstimate.ax", "stateEstimate.ay", "stateEstimate.az"]),
    "quaternion": (40, ["stateEstimate.qx", "stateEstimate.qy", "stateEstimate.qz", "stateEstimate.qw"]),
    "attitude": (40, ["stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw", "stabilizer.thrust"]),
    "imu": (40, ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]),
    "motors": (40, ["motor.m1", "motor.m2", "motor.m3", "motor.m4"]),
    "motor_req": (40, ["motor.m1req", "motor.m2req", "motor.m3req", "motor.m4req"]),
    "mellinger_cmd": (40, ["ctrlMel.cmd_roll", "ctrlMel.cmd_pitch", "ctrlMel.cmd_yaw", "ctrlMel.cmd_thrust"]),
    "mellinger_state": (40, ["ctrlMel.r_roll", "ctrlMel.r_pitch", "ctrlMel.r_yaw", "ctrlMel.accelz", "ctrlMel.zdx", "ctrlMel.zdy"]),
    "mellinger_error": (40, ["ctrlMel.zdz", "ctrlMel.i_err_x", "ctrlMel.i_err_y", "ctrlMel.i_err_z"]),
    #"target_pos": (100, ["ctrltarget.x", "ctrltarget.y", "ctrltarget.z", "ctrltarget.vx", "ctrltarget.vy", "ctrltarget.vz"]),
    #"target_acc": (100, ["ctrltarget.ax", "ctrltarget.ay", "ctrltarget.az", "ctrltarget.roll", "ctrltarget.pitch", "ctrltarget.yaw"]),
    #"power": (200, ["ctrltarget.thrust", "pm.vbat", "pm.batteryLevel", "pm.state", "pm.chargeCurrent"]),
    #"status": (200, ["radio.rssi", "radio.isConnected", "radio.numRxBc", "radio.numRxUc", "supervisor.info", "supervisor.accNorm"]),
    #"environment": (200, ["baro.asl", "baro.pressure", "baro.temp", "mag.x", "mag.y", "mag.z"]),
}
BASE_FIELDS = [
    "time_s", "unix_time_s", "phase", "phase_id", "odom_received",
    "ref_x", "ref_y", "ref_z", "ref_vx", "ref_vy", "ref_vz", "ref_ax", "ref_ay", "ref_az",
    "ref_qx", "ref_qy", "ref_qz", "ref_qw", "ref_roll_rate", "ref_pitch_rate", "ref_yaw_rate",
    "opti_x", "opti_y", "opti_z", "opti_vx", "opti_vy", "opti_vz",
    "opti_qx", "opti_qy", "opti_qz", "opti_qw", "opti_roll_deg", "opti_pitch_deg", "opti_yaw_deg",
    "opti_wx", "opti_wy", "opti_wz",
]

class CrazyflieHover(Node):
    def __init__(self):
        super().__init__("crazyflie_hover")
        #self.uri = "radio://0/80/2M/E7E7E7E7E7" #SMALL Drone
        self.uri = "radio://0/80/2M/E7E7E7EA03"
        self.odom_topic = "optitrack/odom2"
        self.hover_z = 1.2
        self.hover_x = 0.1
        self.hover_y = 0.00
        self.landing_x = -0.4
        self.landing_y = 0.0
        self.hover_duration = 110.0
        self.takeoff_time = 2.5
        self.move_time = 2.0
        self.landing_time = 4.0
        self.landing_settle_time = 0.5
        self.extpos_rate = 100.0
        self.command_rate = 100.0
        self.record_rate = 50.0
        self.takeoff_settle_time = 20.0
        self.lock = threading.Lock()
        self.cf_lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        self.data = {name: math.nan for name in BASE_FIELDS if name not in ("time_s", "phase", "phase_id", "odom_received")}
        self.data.update({"opti_x": 0.0, "opti_y": 0.0, "opti_z": 0.0})
        self.phase, self.phase_id = "waiting", 0
        self.odom_received = False
        self.odom_count = 0
        self.estimator_reset_started = False
        self.mission_started = False
        self.log_configs = []
        self.logged_vars = []
        self.csv_file = None
        self.rows = 0
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        cflib.crtp.init_drivers()
        self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache="./cache"))
        self.get_logger().info(f"Connecting to {self.uri}...")
        self.scf.open_link()
        self.cf = self.scf.cf
        self.cf.param.set_value("stabilizer.estimator", "2")
        time.sleep(0.2)
        self.cf.param.set_value("stabilizer.controller", "2")
        time.sleep(0.2)
        self.start_cf_logging()
        self.csv_path = f"crazyflie_lower_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=BASE_FIELDS + self.logged_vars)
        self.writer.writeheader()
        self.t0 = time.perf_counter()
        self.cf.platform.send_arming_request(True)
        time.sleep(1.0)
        self.create_timer(1.0 / self.extpos_rate, self.send_external_position)
        self.create_timer(1.0 / self.record_rate, self.write_csv)
        self.get_logger().info(f"Connected. Recording {len(self.logged_vars)} Crazyflie signals to {self.csv_path}")
        self.get_logger().info("Waiting for OptiTrack...")

    def log_variable_exists(self, name):
        group, variable = name.split(".", 1)
        return group in self.cf.log.toc.toc and variable in self.cf.log.toc.toc[group]

    def start_cf_logging(self):
        for block, (period_ms, candidates) in LOG_BLOCKS.items():
            variables = [name for name in candidates if self.log_variable_exists(name)]
            if not variables:
                continue
            config = LogConfig(name=block, period_in_ms=period_ms)
            for name in variables:
                config.add_variable(name)
            self.cf.log.add_config(config)
            config.data_received_cb.add_callback(self.cf_log_callback)
            config.error_cb.add_callback(self.cf_log_error)
            config.start()
            self.log_configs.append(config)
            self.logged_vars.extend(variables)

    def cf_log_callback(self, timestamp, data, logconf):
        with self.lock:
            self.data.update(data)

    def cf_log_error(self, logconf, message):
        self.get_logger().error(f"{logconf.name}: {message}")

    @staticmethod
    def quaternion_to_euler(qx, qy, qz, qw):
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def odom_callback(self, msg):
        p, q, v, w = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear, msg.twist.twist.angular
        roll, pitch, yaw = self.quaternion_to_euler(q.x, q.y, q.z, q.w)
        with self.lock:
            self.data.update({"opti_x": p.x, "opti_y": p.y, "opti_z": p.z, "opti_vx": v.x, "opti_vy": v.y, "opti_vz": v.z, "opti_qx": q.x, "opti_qy": q.y, "opti_qz": q.z, "opti_qw": q.w, "opti_roll_deg": roll, "opti_pitch_deg": pitch, "opti_yaw_deg": yaw, "opti_wx": w.x, "opti_wy": w.y, "opti_wz": w.z})
            self.odom_received = True
            self.odom_count += 1

    def get_position(self):
        with self.lock:
            return self.data["opti_x"], self.data["opti_y"], self.data["opti_z"]

    def send_external_position(self):
        with self.lock:
            if not self.odom_received:
                return
            x, y, z, count = self.data["opti_x"], self.data["opti_y"], self.data["opti_z"], self.odom_count
        with self.cf_lock:
            self.cf.extpos.send_extpos(x, y, z)
        if count >= 50 and not self.estimator_reset_started:
            self.estimator_reset_started = True
            threading.Thread(target=self.initialize_estimator, daemon=True).start()

    def initialize_estimator(self):
        self.get_logger().info("Resetting Kalman estimator...")
        reset_estimator(self.scf)
        self.get_logger().info("Kalman estimator ready.")
        if not self.mission_started:
            self.mission_started = True
            threading.Thread(target=self.hover_mission, daemon=True).start()

    def set_phase(self, name, number):
        with self.lock:
            self.phase, self.phase_id = name, number

    def send_setpoint(self, x, y, z, vx=0.0, vy=0.0, vz=0.0, ax=0.0, ay=0.0, az=0.0):
        quaternion = [0.0, 0.0, 0.0, 1.0]
        with self.lock:
            self.data.update({"ref_x": x, "ref_y": y, "ref_z": z, "ref_vx": vx, "ref_vy": vy, "ref_vz": vz, "ref_ax": ax, "ref_ay": ay, "ref_az": az, "ref_qx": 0.0, "ref_qy": 0.0, "ref_qz": 0.0, "ref_qw": 1.0, "ref_roll_rate": 0.0, "ref_pitch_rate": 0.0, "ref_yaw_rate": 0.0})
        with self.cf_lock:
            self.cf.commander.send_full_state_setpoint([x, y, z], [vx, vy, vz], [ax, ay, az], quaternion, 0.0, 0.0, 0.0)

    @staticmethod
    def vertical_reference(z0, zf, t, duration):
        s = max(0.0, min(t / duration, 1.0))
        p = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        dp = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
        ddp = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
        return z0 + (zf - z0) * p, (zf - z0) * dp, (zf - z0) * ddp

    def run_vertical_motion(self, x, y, z0, zf, duration):
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < duration:
            z, vz, az = self.vertical_reference(z0, zf, time.perf_counter() - start, duration)
            self.send_setpoint(x, y, z, vz=vz, az=az)
            time.sleep(1.0 / self.command_rate)
        self.send_setpoint(x, y, zf)

    def run_horizontal_motion(self, x0, y0, xf, yf, z, duration):
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
        self.get_logger().info(f"Taking off to {self.hover_z:.2f} m...")
        self.run_vertical_motion(x0, y0, z0, self.hover_z, self.takeoff_time)
        self.set_phase("takeoff_settle", 2)
        self.get_logger().info(f"Stabilizing for {self.takeoff_settle_time:.1f} seconds...")
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < self.takeoff_settle_time:
            self.send_setpoint(x0, y0, self.hover_z)
            time.sleep(1.0 / self.command_rate)
        self.set_phase("move_to_hover", 3)
        self.run_horizontal_motion(x0, y0, self.hover_x, self.hover_y, self.hover_z, self.move_time)
        self.set_phase("hover", 4)
        self.get_logger().info(f"Hovering for {self.hover_duration:.1f} seconds...")
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < self.hover_duration:
            self.send_setpoint(self.hover_x, self.hover_y, self.hover_z)
            time.sleep(1.0 / self.command_rate)
        self.set_phase("return_to_origin", 5)
        self.run_horizontal_motion(self.hover_x, self.hover_y, self.landing_x, self.landing_y, self.hover_z, self.move_time)
        self.set_phase("landing", 6)
        self.get_logger().info("Landing...")
        self.run_vertical_motion(self.landing_x, self.landing_y, self.hover_z, z0, self.landing_time)
        start = time.perf_counter()
        while not self.shutdown_flag.is_set() and time.perf_counter() - start < self.landing_settle_time:
            self.send_setpoint(self.landing_x, self.landing_y, z0)
            time.sleep(1.0 / self.command_rate)
        self.set_phase("landed", 7)
        self.stop_motors()
        self.get_logger().info(f"Landed. Data saved to {self.csv_path}")
        self.shutdown_flag.set()

    def write_csv(self):
        if self.csv_file is None:
            return
        with self.lock:
            row = {name: self.data.get(name, "") for name in BASE_FIELDS + self.logged_vars}
            row.update({"time_s": time.perf_counter() - self.t0, "unix_time_s": time.time(), "phase": self.phase, "phase_id": self.phase_id, "odom_received": int(self.odom_received)})
        self.writer.writerow(row)
        self.rows += 1
        if self.rows % int(self.record_rate) == 0:
            self.csv_file.flush()

    def stop_motors(self):
        with self.cf_lock:
            self.cf.commander.send_stop_setpoint()
            self.cf.platform.send_arming_request(False)

    def destroy_node(self):
        self.shutdown_flag.set()
        try:
            self.stop_motors()
        except Exception:
            pass
        for config in self.log_configs:
            try:
                config.stop()
            except Exception:
                pass
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
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
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()