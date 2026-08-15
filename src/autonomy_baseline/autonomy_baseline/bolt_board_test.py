#!/usr/bin/env python3

import time
import argparse

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


def set_param(cf, name, value, delay=0.05):
    cf.param.set_value(name, str(value))
    time.sleep(delay)


def stop_all_motors(cf):
    for i in range(1, 5):
        set_param(cf, f"motorPowerSet.m{i}", 0, delay=0.02)

    set_param(cf, "motorPowerSet.enable", 0, delay=0.05)


def test_single_motor(cf, motor_id, power, duration):
    assert motor_id in [1, 2, 3, 4]
    assert 0 <= power <= 65535

    print("[INFO] Stopping all motors...")
    stop_all_motors(cf)

    print("[INFO] Enabling direct motor override...")
    set_param(cf, "motorPowerSet.enable", 1, delay=0.2)

    for i in range(1, 5):
        set_param(cf, f"motorPowerSet.m{i}", 0, delay=0.02)

    motor_name = f"motorPowerSet.m{motor_id}"

    print(f"[INFO] Testing motor M{motor_id} at power={power} for {duration:.2f} s")

    # Small safety ramp
    steps = 10
    for k in range(1, steps + 1):
        cmd = int(power * k / steps)
        set_param(cf, motor_name, cmd, delay=0.1)

    time.sleep(duration)

    print("[INFO] Ramping down...")
    for k in range(steps, -1, -1):
        cmd = int(power * k / steps)
        set_param(cf, motor_name, cmd, delay=0.05)

    stop_all_motors(cf)
    print("[INFO] Done. Motor override disabled.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        default="radio://0/90/2M/E7E7E7E701",
        help="Crazyflie URI, e.g. radio://0/90/2M/E7E7E7E701 or usb://0",
    )
    parser.add_argument("--motor", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--power", type=int, default=3000)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()

    cflib.crtp.init_drivers()

    print(f"[INFO] Connecting to {args.uri}")

    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf

        try:
            test_single_motor(
                cf=cf,
                motor_id=args.motor,
                power=args.power,
                duration=args.duration,
            )
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted. Stopping motors...")
            stop_all_motors(cf)
        except Exception:
            print("[ERROR] Exception occurred. Stopping motors...")
            stop_all_motors(cf)
            raise


if __name__ == "__main__":
    main()