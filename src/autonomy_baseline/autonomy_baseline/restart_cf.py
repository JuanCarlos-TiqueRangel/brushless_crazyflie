#!/usr/bin/env python3
import time
import argparse

import cflib.crtp
from cflib.utils.power_switch import PowerSwitch


def restart_crazyflie(uri: str, mode: str = "stm", wait_after: float = 5.0) -> None:
    """
    Restart a Crazyflie remotely using Crazyradio.

    mode:
        - "stm"        : restart STM MCU + attached decks
        - "fw"         : reboot and start in firmware mode
        - "bootloader" : reboot and start in bootloader mode
    """
    cflib.crtp.init_drivers(enable_debug_driver=False)

    switch = PowerSwitch(uri)
    try:
        if mode == "stm":
            print(f"[INFO] Restarting STM on {uri} ...")
            switch.stm_power_cycle()

        elif mode == "fw":
            print(f"[INFO] Rebooting Crazyflie to firmware on {uri} ...")
            switch.reboot_to_fw()

        elif mode == "bootloader":
            print(f"[INFO] Rebooting Crazyflie to bootloader on {uri} ...")
            switch.reboot_to_bootloader()

        else:
            raise ValueError(f"Unknown mode: {mode}")

    finally:
        switch.close()

    time.sleep(wait_after)
    print(f"[OK] Command sent. Waited {wait_after:.1f} s for reboot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restart Crazyflie remotely")
    parser.add_argument(
        "--uri",
        type=str,
        default="radio://0/80/2M/E7E7E7E705",
        help="Crazyflie radio URI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stm", "fw", "bootloader"],
        default="stm",
        help="Restart mode"
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=5.0,
        help="Seconds to wait after sending reboot command"
    )

    args = parser.parse_args()
    restart_crazyflie(args.uri, args.mode, args.wait)