"""Stream live joint positions from the SO-101 leader arm (no follower needed).

Usage:
    python scripts/leader_stream_test.py --port /dev/ttyACM0 --id cobalt_leader_arm

Move the arm and watch the values update. Ctrl+C to stop.
"""

import argparse
import time

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="/dev/ttyACM0")
parser.add_argument("--id", default="cobalt_leader_arm")
args = parser.parse_args()

config = SO101LeaderConfig(port=args.port, id=args.id)
leader = SO101Leader(config)
leader.connect(calibrate=False)

try:
    while True:
        action = leader.get_action()
        line = "  ".join(f"{k}={v:7.2f}" for k, v in action.items())
        print(line, end="\r")
        time.sleep(0.05)
except KeyboardInterrupt:
    print()
finally:
    leader.disconnect()
