import time
import logging

import numpy as np
from scipy.spatial.transform import Rotation

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from NatNetClient import NatNetClient

# =============================================================================
# HARDCODED CONFIG
# =============================================================================
RADIO_CHANNEL = 80
RADIO_DATARATE = "2M"
RADIO_ADDRESS = "E7E7E7E705"

CLIENT_IP = "192.168.0.35"
OPTITRACK_SERVER_IP = "192.168.0.4"
RIGID_BODY_ID = 527

HOVER_HEIGHT_ABOVE_START = 0.5   # meters
COMMAND_PERIOD = 0.02            # 50 Hz
OPTITRACK_TIMEOUT = 15.0         # seconds

# =============================================================================
# GLOBAL STATE
# =============================================================================
positions = {}
rotations = {}
uri = None
latest_console_line = False


# =============================================================================
# CALLBACKS
# =============================================================================
def console_callback(text: str):
    """
    Called when text is printed from the Crazyflie firmware console.
    """
    global latest_console_line
    print(text, end="")
    latest_console_line = True


def receive_rigid_body_frame(robot_id, position, rotation_quaternion):
    """
    Receive rigid body data from OptiTrack.
    NatNet quaternion is typically (x, y, z, w) in your current pipeline.
    """
    positions[robot_id] = np.array(position[:3], dtype=float)
    rotations[robot_id] = np.array(rotation_quaternion[:4], dtype=float)


def param_callback(name, value):
    print(f"The crazyflie has parameter {name} set to: {value}")


def set_param(cf, groupstr, namestr, value):
    full_name = f"{groupstr}.{namestr}"
    cf.param.add_update_callback(group=groupstr, name=namestr, cb=param_callback)
    cf.param.set_value(full_name, value)


def _connected(link_uri):
    print(f"Connected to {link_uri}")


def _connection_failed(link_uri, msg):
    print(f"Connection to {link_uri} failed: {msg}")


def _connection_lost(link_uri, msg):
    print(f"Connection to {link_uri} lost: {msg}")


def _disconnected(link_uri):
    print(f"Disconnected from {link_uri}")


# =============================================================================
# INIT
# =============================================================================
def initialize_optitrack():
    """
    Start NatNet/OptiTrack streaming and wait for connection.
    """
    streaming_client = NatNetClient()
    streaming_client.set_client_address(CLIENT_IP)
    streaming_client.set_server_address(OPTITRACK_SERVER_IP)
    streaming_client.set_use_multicast(True)
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    is_running = streaming_client.run()
    time.sleep(2.0)

    if is_running and streaming_client.connected():
        print("Connected to OptiTrack")
        return streaming_client
    raise RuntimeError("Could not connect to OptiTrack")


def initialize_crazyflie():
    """
    Connect to Crazyflie and attach console callback.
    """
    global uri

    uri_address = f"radio://0/{RADIO_CHANNEL}/{RADIO_DATARATE}/{RADIO_ADDRESS}"
    uri = uri_helper.uri_from_env(default=uri_address)

    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache="./cache")

    cf.connected.add_callback(_connected)
    cf.connection_failed.add_callback(_connection_failed)
    cf.connection_lost.add_callback(_connection_lost)
    cf.disconnected.add_callback(_disconnected)

    # Firmware console callback
    cf.console.receivedChar.add_callback(console_callback)

    print(f"Connecting to Crazyflie at {uri}")
    print(
        f"Client IP: {CLIENT_IP} | OptiTrack server: {OPTITRACK_SERVER_IP} | "
        f"Rigid body ID: {RIGID_BODY_ID}"
    )

    cf.open_link(uri)
    time.sleep(2.0)
    return cf


# =============================================================================
# HELPERS
# =============================================================================
def wait_for_optitrack_pose(rigid_body_id, timeout_s=15.0):
    """
    Wait until at least one OptiTrack pose arrives.
    """
    print(f"Waiting for OptiTrack pose for rigid body {rigid_body_id}...")
    t0 = time.time()
    while rigid_body_id not in positions:
        if (time.time() - t0) > timeout_s:
            raise RuntimeError(
                f"No OptiTrack data for rigid body {rigid_body_id} after {timeout_s:.1f} s"
            )
        time.sleep(0.05)
    print("OptiTrack pose received.")


def get_optitrack_pose(rigid_body_id):
    """
    Returns:
        pos: np.array([x, y, z])
        quat_xyzw: np.array([qx, qy, qz, qw])
        yaw_deg: float
    """
    pos = np.array(positions[rigid_body_id], dtype=float)
    quat_xyzw = np.array(rotations[rigid_body_id], dtype=float)

    # scipy Rotation.from_quat expects [x, y, z, w]
    rot = Rotation.from_quat(quat_xyzw)
    roll_deg, pitch_deg, yaw_deg = rot.as_euler("xyz", degrees=True)

    return pos, quat_xyzw, roll_deg, pitch_deg, yaw_deg


def reset_kalman_estimator(cf):
    """
    Reset Kalman estimator after setting initial state.
    """
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)


def send_external_pose(cf, pos, quat_xyzw):
    """
    Send external pose to Crazyflie localization module.

    cflib supports send_extpose(pos, quat) where:
      pos  = [x, y, z]
      quat = [qx, qy, qz, qw]
    and this is forwarded to the Crazyflie position estimator. :contentReference[oaicite:1]{index=1}
    """
    cf.loc.send_extpose(pos, quat_xyzw)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    streaming_client = None
    cf = None

    try:
        # ---------------------------------------------------------------------
        # Initialize systems
        # ---------------------------------------------------------------------
        streaming_client = initialize_optitrack()
        cf = initialize_crazyflie()

        wait_for_optitrack_pose(RIGID_BODY_ID, timeout_s=OPTITRACK_TIMEOUT)

        # ---------------------------------------------------------------------
        # Read initial pose from OptiTrack
        # ---------------------------------------------------------------------
        p_r, q_r_xyzw, roll_r, pitch_r, yaw_r = get_optitrack_pose(RIGID_BODY_ID)

        print(
            "Initial OptiTrack pose:\n"
            f"  position = [{p_r[0]:.3f}, {p_r[1]:.3f}, {p_r[2]:.3f}] m\n"
            f"  attitude = roll {roll_r:.2f} deg, pitch {pitch_r:.2f} deg, yaw {yaw_r:.2f} deg"
        )

        # ---------------------------------------------------------------------
        # Use OptiTrack pose for state estimation
        # ---------------------------------------------------------------------
        # Set Kalman estimator
        set_param(cf, "stabilizer", "estimator", 2)   # 2 = Kalman
        time.sleep(0.2)

        # Use the default PID controller for standard hover
        set_param(cf, "stabilizer", "controller", 1)  # 1 = PID
        time.sleep(0.2)

        # Push a few pose packets before reset so estimator sees external pose
        for _ in range(20):
            p_r, q_r_xyzw, _, _, yaw_r = get_optitrack_pose(RIGID_BODY_ID)
            send_external_pose(cf, p_r, q_r_xyzw)
            time.sleep(0.02)

        # Initialize Kalman state from current OptiTrack pose
        cf.param.set_value("kalman.initialX", str(float(p_r[0])))
        cf.param.set_value("kalman.initialY", str(float(p_r[1])))
        cf.param.set_value("kalman.initialZ", str(float(p_r[2])))
        cf.param.set_value("kalman.initialYaw", str(float(np.deg2rad(yaw_r))))
        reset_kalman_estimator(cf)

        print("Kalman estimator initialized from OptiTrack pose.")

        # ---------------------------------------------------------------------
        # Target: hover above starting position
        # ---------------------------------------------------------------------
        target_x = float(p_r[0])
        target_y = float(p_r[1])
        target_z = float(p_r[2] + HOVER_HEIGHT_ABOVE_START)
        target_yaw_deg = float(yaw_r)

        print(
            f"Target setpoint: "
            f"[{target_x:.3f}, {target_y:.3f}, {target_z:.3f}] m, yaw {target_yaw_deg:.2f} deg"
        )

        # Unlock thrust protection
        cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
        time.sleep(0.1)

        # ---------------------------------------------------------------------
        # Main hover loop
        # ---------------------------------------------------------------------
        print("Starting hover loop. Press Ctrl+C to stop.")

        last_print = time.time()

        while True:
            loop_start = time.time()

            # Read fresh OptiTrack pose
            p_r, q_r_xyzw, roll_r, pitch_r, yaw_r = get_optitrack_pose(RIGID_BODY_ID)

            # Feed external pose into estimator continuously
            send_external_pose(cf, p_r, q_r_xyzw)

            # Send default position hover command
            cf.commander.send_position_setpoint(
                target_x, target_y, target_z, target_yaw_deg
            )

            # Optional status print at ~2 Hz
            if (time.time() - last_print) > 0.5:
                last_print = time.time()
                print(
                    f"Pose: x={p_r[0]:.3f}, y={p_r[1]:.3f}, z={p_r[2]:.3f} | "
                    f"roll={roll_r:.1f}, pitch={pitch_r:.1f}, yaw={yaw_r:.1f}"
                )

            # Maintain loop period
            dt = time.time() - loop_start
            sleep_time = COMMAND_PERIOD - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Landing/stopping...")

        if cf is not None:
            try:
                # Graceful stop
                for _ in range(20):
                    cf.commander.send_position_setpoint(
                        target_x, target_y, p_r[2], target_yaw_deg
                    )
                    time.sleep(0.02)

                cf.commander.send_stop_setpoint()
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during shutdown: {e}")

    except Exception as e:
        print(f"Fatal error: {e}")

    finally:
        if cf is not None:
            try:
                cf.close_link()
            except Exception:
                pass

        if streaming_client is not None:
            try:
                streaming_client.shutdown()
            except Exception:
                pass

        print("Clean exit.")
