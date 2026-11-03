"""Assign the correct ID/baudrate to ONE replacement motor on the SO-101 follower.

Avoids walking all 6 motors like `lerobot-setup-motors`. Connect ONLY the new motor
to the follower's controller board (unplug the daisy chain), then run:

    conda activate so_arm101
    python scripts/setup_one_motor.py elbow_flex      # FOLLOWER_PORT / FOLLOWER_ID env vars override the defaults

Joint names: shoulder_pan(1) shoulder_lift(2) elbow_flex(3) wrist_flex(4) wrist_roll(5) gripper(6)
"""

import os
import sys

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E089183-if00")

robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=os.environ.get("FOLLOWER_ID", "neon_green_follower_arm")))
motors = list(robot.bus.motors)

if len(sys.argv) != 2 or sys.argv[1] not in motors:
    sys.exit(f"usage: python {sys.argv[0]} <joint>   (one of: {', '.join(motors)})")

motor = sys.argv[1]
input(f"Connect ONLY the new '{motor}' motor to the follower board (with power on), then press Enter.")
robot.bus.setup_motor(motor)
print(f"'{motor}' motor id set to {robot.bus.motors[motor].id}")
