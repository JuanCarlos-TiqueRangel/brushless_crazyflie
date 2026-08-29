#!/usr/bin/env python3

import threading
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.reset_estimator import reset_estimator


class CrazyflieHover(Node):

    def __init__(self):
        super().__init__("crazyflie_hover")

        # --------------------------------------------------
        # SETTINGS
        # --------------------------------------------------
        # self.uri = "radio://0/80/2M/E7E7E7EA01"
        self.uri = "radio://1/90/2M/E7E7E7EA06"
        self.odom_topic = "optitrack/odom"

        self.hover_z = 0.5          # meters
        self.hover_duration = 240.0  # seconds

        self.extpos_rate = 100.0    # Hz
        self.command_rate = 100.0   # Hz

        # --------------------------------------------------
        # OptiTrack state
        # --------------------------------------------------
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.odom_received = False
        self.odom_count = 0

        self.lock = threading.Lock()
        self.cf_lock = threading.Lock()

        self.estimator_ready = False
        self.estimator_reset_started = False
        self.mission_started = False

        self.shutdown_flag = threading.Event()

        # --------------------------------------------------
        # OptiTrack subscriber
        # --------------------------------------------------
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        # --------------------------------------------------
        # Connect Crazyflie
        # --------------------------------------------------
        cflib.crtp.init_drivers()

        self.scf = SyncCrazyflie(
            self.uri,
            cf=Crazyflie(rw_cache="./cache")
        )

        self.get_logger().info(f"Connecting to {self.uri}...")

        self.scf.open_link()
        self.cf = self.scf.cf

        self.get_logger().info("Crazyflie connected.")

        # Kalman estimator
        self.cf.param.set_value("stabilizer.estimator", "2")
        time.sleep(0.2)

        # Mellinger controller
        self.cf.param.set_value("stabilizer.controller", "2")
        time.sleep(0.2)

        self.get_logger().info("Kalman estimator + Mellinger controller enabled.")

        # Arm
        self.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        # --------------------------------------------------
        # Continuously send OptiTrack position
        # --------------------------------------------------
        self.create_timer(
            1.0 / self.extpos_rate,
            self.send_external_position,
        )

        self.get_logger().info("Waiting for OptiTrack...")


    # ==================================================
    # OPTITRACK
    # ==================================================
    def odom_callback(self, msg):

        with self.lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            self.z = msg.pose.pose.position.z

            self.odom_received = True
            self.odom_count += 1


    def get_position(self):

        with self.lock:
            return self.x, self.y, self.z


    # ==================================================
    # SEND OPTITRACK TO CRAZYFLIE
    # ==================================================
    def send_external_position(self):

        if not self.odom_received:
            return

        x, y, z = self.get_position()

        with self.cf_lock:
            self.cf.extpos.send_extpos(x, y, z)

        # Wait for some OptiTrack samples before resetting Kalman
        if (
            self.odom_count >= 50
            and not self.estimator_reset_started
        ):
            self.estimator_reset_started = True

            threading.Thread(
                target=self.initialize_estimator,
                daemon=True,
            ).start()


    # ==================================================
    # RESET KALMAN
    # ==================================================
    def initialize_estimator(self):

        self.get_logger().info("Resetting Kalman estimator...")

        reset_estimator(self.scf)

        self.estimator_ready = True

        self.get_logger().info("Kalman estimator ready.")

        if not self.mission_started:

            self.mission_started = True

            threading.Thread(
                target=self.hover_mission,
                daemon=True,
            ).start()


    # ==================================================
    # SEND FULL STATE
    # ==================================================
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

        # yaw = 0
        quaternion = [0.0, 0.0, 0.0, 1.0]

        with self.cf_lock:

            self.cf.commander.send_full_state_setpoint(
                [x, y, z],
                [vx, vy, vz],
                [ax, ay, az],
                quaternion,
                0.0,
                0.0,
                0.0,
            )


    # ==================================================
    # SMOOTH VERTICAL MOTION
    # ==================================================
    def vertical_reference(self, x, y, z0, zf, t, T):

        s = max(0.0, min(t / T, 1.0))

        # Quintic smooth step
        p = (
            10.0 * s**3
            - 15.0 * s**4
            + 6.0 * s**5
        )

        dp = (
            30.0 * s**2
            - 60.0 * s**3
            + 30.0 * s**4
        ) / T

        ddp = (
            60.0 * s
            - 180.0 * s**2
            + 120.0 * s**3
        ) / (T * T)

        dz = zf - z0

        z = z0 + dz * p
        vz = dz * dp
        az = dz * ddp

        return z, vz, az


    # ==================================================
    # HOVER MISSION
    # ==================================================
    def hover_mission(self):

        dt = 1.0 / self.command_rate

        # Current OptiTrack position
        x0, y0, z0 = self.get_position()

        self.get_logger().info(
            f"Initial position: "
            f"x={x0:.2f}, y={y0:.2f}, z={z0:.2f}"
        )

        # Stop previous commander setpoints
        with self.cf_lock:
            self.cf.commander.send_notify_setpoint_stop()

        # ==================================================
        # TAKEOFF
        # ==================================================
        takeoff_time = 2.5

        self.get_logger().info(
            f"Taking off to {self.hover_z:.2f} m..."
        )

        start = time.time()

        while time.time() - start < takeoff_time:

            t = time.time() - start

            z_ref, vz_ref, az_ref = self.vertical_reference(
                x0,
                y0,
                z0,
                self.hover_z,
                t,
                takeoff_time,
            )

            self.send_setpoint(
                x0,
                y0,
                z_ref,
                vz=vz_ref,
                az=az_ref,
            )

            time.sleep(dt)

        # ==================================================
        # HOVER
        # ==================================================
        self.get_logger().info(
            f"Hovering for {self.hover_duration:.1f} seconds..."
        )

        start = time.time()

        while time.time() - start < self.hover_duration:

            self.send_setpoint(
                x0,
                y0,
                self.hover_z,
            )

            time.sleep(dt)

        # ==================================================
        # LAND
        # ==================================================
        landing_time = 2.5

        self.get_logger().info("Landing...")

        start = time.time()

        while time.time() - start < landing_time:

            t = time.time() - start

            z_ref, vz_ref, az_ref = self.vertical_reference(
                x0,
                y0,
                self.hover_z,
                z0,
                t,
                landing_time,
            )

            self.send_setpoint(
                x0,
                y0,
                z_ref,
                vz=vz_ref,
                az=az_ref,
            )

            time.sleep(dt)

        self.get_logger().info("Landed.")

        self.stop_motors()


    # ==================================================
    # STOP
    # ==================================================
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

        try:
            self.scf.close_link()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CrazyflieHover()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()