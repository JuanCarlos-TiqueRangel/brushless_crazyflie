#!/usr/bin/env python3
import math
import threading
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import csv
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.reset_estimator import reset_estimator

@dataclass
class CrazyflieConfig:
    # Select controller
    # '1')  # PID
    # '3')  # INDI
    controller: str = "3"
    controller_str: str = "INDI"


class CrazyflieFigure8Stage1(Node):
    def __init__(self) -> None:
        super().__init__('crazyflie_figure8_stage1')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('uri', 'radio://0/80/2M/E7E7E7E705')
        self.declare_parameter('odom_topic', 'optitrack/odom')

        self.declare_parameter('extpos_rate_hz', 100.0)
        self.declare_parameter('odom_timeout_s', 0.20)
        self.declare_parameter('estimator_reset_samples', 50)

        self.declare_parameter('auto_start', True)
        self.declare_parameter('arm_on_start', True)

        self.declare_parameter('use_current_position_as_center', True)
        self.declare_parameter('center_x', 0.0)
        self.declare_parameter('center_y', 0.0)

        self.declare_parameter('hover_z', 0.6)
        self.declare_parameter('takeoff_duration_s', 2.5)
        self.declare_parameter('land_duration_s', 2.5)
        self.declare_parameter('move_to_start_duration_s', 2.0)

        self.declare_parameter('radius_x', 2.0)
        self.declare_parameter('radius_y', 1.0)
        self.declare_parameter('num_points_per_loop', 40)
        self.declare_parameter('num_loops', 4)
        self.declare_parameter('loop_duration_s', 8.0)

        self.declare_parameter('yaw_rad', 0.0)
        self.declare_parameter('segment_margin_s', 0.05)
        self.declare_parameter('start_delay_after_reset_s', 1.0)

        self.uri = self.get_parameter('uri').value
        self.odom_topic = self.get_parameter('odom_topic').value

        self.extpos_rate_hz = float(self.get_parameter('extpos_rate_hz').value)
        self.odom_timeout_s = float(self.get_parameter('odom_timeout_s').value)
        self.estimator_reset_samples = int(self.get_parameter('estimator_reset_samples').value)

        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.arm_on_start = bool(self.get_parameter('arm_on_start').value)

        self.use_current_position_as_center = bool(
            self.get_parameter('use_current_position_as_center').value
        )
        self.center_x_param = float(self.get_parameter('center_x').value)
        self.center_y_param = float(self.get_parameter('center_y').value)

        self.hover_z = float(self.get_parameter('hover_z').value)
        self.takeoff_duration_s = float(self.get_parameter('takeoff_duration_s').value)
        self.land_duration_s = float(self.get_parameter('land_duration_s').value)
        self.move_to_start_duration_s = float(
            self.get_parameter('move_to_start_duration_s').value
        )

        self.radius_x = float(self.get_parameter('radius_x').value)
        self.radius_y = float(self.get_parameter('radius_y').value)
        self.num_points_per_loop = int(self.get_parameter('num_points_per_loop').value)
        self.num_loops = int(self.get_parameter('num_loops').value)
        self.loop_duration_s = float(self.get_parameter('loop_duration_s').value)

        self.yaw_rad = float(self.get_parameter('yaw_rad').value)
        self.segment_margin_s = float(self.get_parameter('segment_margin_s').value)
        self.start_delay_after_reset_s = float(
            self.get_parameter('start_delay_after_reset_s').value
        )

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

        # -----------------------------
        # ROS subscriber
        # -----------------------------
        self.sub_odom = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        # -----------------------------
        # Data collection
        # -----------------------------
        self.declare_parameter('enable_csv_logging', True)
        self.declare_parameter('log_rate_hz', 20.0)

        self.enable_csv_logging = bool(self.get_parameter('enable_csv_logging').value)
        self.log_rate_hz = float(self.get_parameter('log_rate_hz').value)

        self.x_ref = 0.0
        self.y_ref = 0.0
        self.z_ref = self.hover_z

        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / 'logs'

        self.csv_file = None
        self.csv_writer = None

        if self.enable_csv_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = str(self.log_dir / f'figure8_log_{CrazyflieConfig.controller_str}.csv')
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                't_wall',
                'x_ref', 'y_ref', 'z_ref',
                'x_meas', 'y_meas', 'z_meas',
                'vx_meas', 'vy_meas', 'vz_meas'
            ])
            self.log_timer = self.create_timer(1.0 / self.log_rate_hz, self.log_timer_callback)
            self.get_logger().info(f'CSV logging to {self.csv_path}')



        # -----------------------------
        # Crazyflie connect
        # -----------------------------
        self.connect_to_crazyflie()

        # Continuously feed external position
        self.extpos_timer = self.create_timer(
            1.0 / self.extpos_rate_hz,
            self.extpos_timer_callback
        )

        self.get_logger().info('Crazyflie Figure-8 Stage 1 node started.')

    # --------------------------------------------------
    # Crazyflie setup
    # --------------------------------------------------
    def connect_to_crazyflie(self) -> None:
        self.get_logger().info(f'Connecting to Crazyflie at {self.uri} ...')
        cflib.crtp.init_drivers()

        self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache'))
        self.scf.open_link()
        self.cf = self.scf.cf
        self.connected = True

        self.get_logger().info('Connected to Crazyflie.')

        # Use Kalman estimator for external position
        try:
            self.cf.param.set_value('stabilizer.estimator', '2')
            time.sleep(0.2)
            self.get_logger().info('Set stabilizer.estimator = 2 (Kalman).')
        except Exception as exc:
            self.get_logger().warn(f'Could not set Kalman estimator: {exc}')

        # Select Controller
        try:
            self.cf.param.set_value('stabilizer.controller', CrazyflieConfig.controller)   # change this number
            time.sleep(0.2)
            self.get_logger().info('Set stabilizer.controller = 1')
        except Exception as exc:
            self.get_logger().warn(f'Could not set controller: {exc}')

        # PID is the default controller, so we leave controller selection untouched

        if self.arm_on_start:
            try:
                self.cf.platform.send_arming_request(True)
                time.sleep(1.0)
                self.get_logger().info('Arming request sent.')
            except Exception as exc:
                self.get_logger().warn(f'Automatic arming failed: {exc}')



    def log_timer_callback(self) -> None:
        if not self.enable_csv_logging or self.csv_writer is None:
            return
        if not self.odom_received:
            return

        x, y, z, vx, vy, vz = self.get_state_copy()

        self.csv_writer.writerow([
            time.time(),
            self.x_ref, self.y_ref, self.z_ref,
            x, y, z,
            vx, vy, vz
        ])

        if self.csv_file is not None:
            self.csv_file.flush()


    # --------------------------------------------------
    # ROS callback
    # --------------------------------------------------
    def odom_callback(self, msg: Odometry) -> None:
        with self.state_lock:
            # Assumes this topic is already in the Crazyflie world frame and meters
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
    # External position feed
    # --------------------------------------------------
    def extpos_timer_callback(self) -> None:
        if not self.connected or self.cf is None:
            return
        if not self.odom_received:
            return

        x, y, z, _, _, _ = self.get_state_copy()

        #self.get_logger().info(f"Sending extpos: {x:.3f}, {y:.3f}, {z:.3f}")

        try:
            with self.cf_lock:
                self.cf.extpos.send_extpos(x, y, z)
        except Exception as exc:
            self.get_logger().error(f'Failed to send extpos: {exc}')
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

        self.get_logger().info('Resetting estimator...')
        try:
            # Do NOT hold cf_lock here, otherwise extpos streaming gets blocked
            reset_estimator(self.scf)
            self.estimator_ready = True
            self.get_logger().info('Estimator is ready.')

            if self.auto_start and not self.mission_started:
                time.sleep(self.start_delay_after_reset_s)
                self.mission_started = True
                threading.Thread(target=self.mission_worker, daemon=True).start()

        except Exception as exc:
            self.get_logger().error(f'Estimator reset failed: {exc}')
            self.estimator_reset_started = False

    # --------------------------------------------------
    # Figure-8 generation
    # --------------------------------------------------
    def build_figure8_points(
        self,
        cx: float,
        cy: float,
        z: float,
        rx: float,
        ry: float,
        n: int
    ) -> List[Tuple[float, float, float]]:
        """
        Figure-8 parameterization:
            x = cx + rx * sin(theta)
            y = cy + ry * sin(theta) * cos(theta)
            z = constant
        """
        pts = []
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            x = cx + rx * math.sin(theta)
            y = cy + ry * math.sin(theta) * math.cos(theta)
            pts.append((x, y, z))
        return pts

    # --------------------------------------------------
    # Mission
    # --------------------------------------------------
    def mission_worker(self) -> None:
        if self.cf is None:
            return

        try:
            self.get_logger().info('Starting Stage 1 figure-8 mission...')

            if not self.odom_is_fresh():
                raise RuntimeError('OptiTrack odometry is stale before mission start.')

            x0, y0, _, _, _, _ = self.get_state_copy()

            # Figure-8 center
            if self.use_current_position_as_center:
                cx = x0
                cy = y0
            else:
                cx = self.center_x_param
                cy = self.center_y_param

            zf = self.hover_z

            # Build figure-8 reference points
            points = self.build_figure8_points(
                cx=cx,
                cy=cy,
                z=zf,
                rx=self.radius_x,
                ry=self.radius_y,
                n=self.num_points_per_loop,
            )

            segment_duration = self.loop_duration_s / float(self.num_points_per_loop)

            hlc = self.cf.high_level_commander

            # Re-enable high-level commander in case any low-level setpoint was sent before
            with self.cf_lock:
                self.cf.commander.send_notify_setpoint_stop()

            # -----------------------------
            # Takeoff
            # -----------------------------
            self.x_ref = cx
            self.y_ref = cy
            self.z_ref = zf

            self.get_logger().info(f'Takeoff to z={zf:.2f} m')
            with self.cf_lock:
                hlc.takeoff(zf, self.takeoff_duration_s)
            time.sleep(self.takeoff_duration_s + 1.0)

            # -----------------------------
            # Move to first point
            # -----------------------------
            x_start, y_start, z_start = points[0]
            self.x_ref = x_start
            self.y_ref = y_start
            self.z_ref = z_start

            self.get_logger().info(
                f'Move to first point: ({x_start:.2f}, {y_start:.2f}, {z_start:.2f})'
            )
            with self.cf_lock:
                hlc.go_to(
                    x_start, y_start, z_start,
                    self.yaw_rad,
                    self.move_to_start_duration_s,
                    relative=False
                )
            time.sleep(self.move_to_start_duration_s + self.segment_margin_s)

            # -----------------------------
            # Figure-8 loops
            # -----------------------------
            for loop_idx in range(self.num_loops):
                self.get_logger().info(f'Figure-8 loop {loop_idx + 1}/{self.num_loops}')

                for px, py, pz in points:
                    if self.shutdown_flag.is_set() or self.mission_abort.is_set():
                        raise RuntimeError('Mission aborted.')

                    if not self.odom_is_fresh():
                        raise RuntimeError('OptiTrack odometry became stale during mission.')

                    # Update reference for CSV logging
                    self.x_ref = px
                    self.y_ref = py
                    self.z_ref = pz

                    with self.cf_lock:
                        hlc.go_to(
                            px, py, pz,
                            self.yaw_rad,
                            segment_duration,
                            relative=False
                        )

                    time.sleep(segment_duration + self.segment_margin_s)

            # -----------------------------
            # Return to center
            # -----------------------------
            self.x_ref = cx
            self.y_ref = cy
            self.z_ref = zf

            self.get_logger().info('Returning to center...')
            with self.cf_lock:
                hlc.go_to(cx, cy, zf, self.yaw_rad, 2.0, relative=False)
            time.sleep(2.2)

            # -----------------------------
            # Land
            # -----------------------------
            self.x_ref = cx
            self.y_ref = cy
            self.z_ref = 0.0

            self.get_logger().info('Landing...')
            with self.cf_lock:
                hlc.land(0.0, self.land_duration_s)
            time.sleep(self.land_duration_s + 1.0)

            self.get_logger().info('Mission completed successfully.')

        except Exception as exc:
            self.get_logger().error(f'Mission error: {exc}')
            self.abort_and_land()

    def abort_and_land(self) -> None:
        if self.cf is None:
            return

        try:
            hlc = self.cf.high_level_commander
            self.get_logger().warn('Abort requested, attempting controlled landing...')
            with self.cf_lock:
                hlc.land(0.0, max(1.5, self.land_duration_s))
            time.sleep(max(1.5, self.land_duration_s) + 0.5)
        except Exception as exc:
            self.get_logger().warn(f'Controlled landing failed: {exc}')
            try:
                with self.cf_lock:
                    self.cf.commander.send_stop_setpoint()
            except Exception:
                pass

    # --------------------------------------------------
    # Shutdown
    # --------------------------------------------------
    def destroy_node(self):
        self.get_logger().info('Shutting down node...')
        self.shutdown_flag.set()
        self.mission_abort.set()

        try:
            self.abort_and_land()
        except Exception:
            pass

        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception as exc:
                self.get_logger().warn(f'Failed to close CSV file: {exc}')
            finally:
                self.csv_file = None
                self.csv_writer = None

        if self.cf is not None:
            try:
                with self.cf_lock:
                    self.cf.commander.send_stop_setpoint()
            except Exception:
                pass

            try:
                with self.cf_lock:
                    self.cf.platform.send_arming_request(False)
            except Exception:
                pass

        if self.scf is not None:
            try:
                self.scf.close_link()
            except Exception as exc:
                self.get_logger().warn(f'close_link failed: {exc}')

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CrazyflieFigure8Stage1()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()