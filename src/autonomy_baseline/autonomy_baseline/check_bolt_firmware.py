#!/usr/bin/env python3

import time
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


URI = "radio://0/90/2M/E7E7E7E701"


def main():
    cflib.crtp.init_drivers()

    params_to_read = [
        "firmware.revision0",
        "firmware.revision1",
        "firmware.modified",
        "system.selftestPassed",
    ]

    values = {}
    events = {}

    print(f"[INFO] Connecting to {URI}")

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf

        print("[INFO] Connected.")

        def make_callback(param_name):
            def cb(name, value):
                values[name] = value
                events[name].set()
            return cb

        for p in params_to_read:
            group, name = p.split(".")
            events[p] = threading.Event()
            cf.param.add_update_callback(
                group=group,
                name=name,
                cb=make_callback(p)
            )

        time.sleep(0.5)

        for p in params_to_read:
            cf.param.request_param_update(p)

        for p in params_to_read:
            events[p].wait(timeout=2.0)

        print("\n========== Firmware / System Info ==========")

        for p in params_to_read:
            print(f"{p}: {values.get(p, 'NOT FOUND')}")

        try:
            r0 = int(values["firmware.revision0"])
            r1 = int(values["firmware.revision1"])
            print(f"\nFirmware revision chunks: {r0:08x}{r1:04x}")
        except Exception:
            pass

        print("============================================")


if __name__ == "__main__":
    main()