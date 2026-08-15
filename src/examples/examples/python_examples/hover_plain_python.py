#!/usr/bin/env python3

import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.reset_estimator import reset_estimator

from NatNetClient import NatNetClient


URI = 'radio://0/80/2M/E7E7E7E705'
CLIENT_IP = '192.168.0.123'
SERVER_IP = '192.168.0.4'
RIGID_BODY_ID = 541

HOVER_HEIGHT = 0.6
TAKEOFF_TIME = 2.5
LAND_TIME = 2.5

cf = None
latest_sample = None
extpos_count = 0


def receive_rigid_body(rigid_body_id, position, rotation):
    global latest_sample, extpos_count

    if rigid_body_id != RIGID_BODY_ID:
        return

    x, y, z = float(position[0]), float(position[1]), float(position[2])
    latest_sample = (x, y, z, time.monotonic())

    if cf is not None:
        cf.extpos.send_extpos(x, y, z)
        extpos_count += 1


optitrack = NatNetClient()
optitrack.set_client_address(CLIENT_IP)
optitrack.set_server_address(SERVER_IP)
optitrack.set_use_multicast(True)
optitrack.rigid_body_listener = receive_rigid_body

if not optitrack.run():
    raise RuntimeError('Could not start OptiTrack streaming.')

cflib.crtp.init_drivers()

try:
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        flying = False
        try:
            cf.param.set_value('stabilizer.estimator', '2')
            cf.param.set_value('stabilizer.controller', '3')
            cf.param.set_value('commander.enHighLevel', '1')

            print('Waiting for OptiTrack...')
            start = time.monotonic()
            while extpos_count < 50:
                if time.monotonic() - start > 10.0:
                    raise RuntimeError('No OptiTrack data received for rigid body 541.')
                time.sleep(0.01)

            reset_estimator(scf)
            cf.platform.send_arming_request(True)
            time.sleep(1.0)

            cf.commander.send_notify_setpoint_stop()
            cf.high_level_commander.takeoff(HOVER_HEIGHT, TAKEOFF_TIME)
            flying = True
            time.sleep(TAKEOFF_TIME)

            print('Hovering. Press Ctrl+C to land.')

            while True:
                if time.monotonic() - latest_sample[3] > 0.2:
                    raise RuntimeError('OptiTrack data stopped.')
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            if flying:
                cf.high_level_commander.land(0.0, LAND_TIME)
                time.sleep(LAND_TIME)
            cf.commander.send_stop_setpoint()
            cf.platform.send_arming_request(False)
            cf = None
finally:
    cf = None
    optitrack.shutdown()