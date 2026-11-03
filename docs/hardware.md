# Hardware setup and lessons

## The setup

- **Arms:** an SO-101 follower with all **STS3215 12 V** motors on a 12 V supply, and a leader arm with 7.4 V motors on
  a 5–6 V supply. Label the two power bricks: they look alike, and the wrong one on the wrong arm is easy to do.
- **Motor IDs** (same on both arms): shoulder_pan 1, shoulder_lift 2, elbow_flex 3, wrist_flex 4, wrist_roll 5, gripper 6.
- **Cameras:** `front` (a generic USB webcam, fixed workspace view) and `wrist` (Innomaker U20CAM), both 640×480 at 30 fps.
- **Computers:** a laptop with an RTX 4070 Laptop GPU drives the arm (teleoperation, recording, evaluation); a desktop
  with an RTX 4080 trains. Data and models move between them through the Hugging Face Hub.

## Ports, cameras and calibration

Use the stable `/dev/serial/by-id/...` and `/dev/v4l/by-id/...` paths instead of `/dev/ttyACM0` or `/dev/video0`,
which change with plug order. List yours with `ls -l /dev/serial/by-id/ /dev/v4l/by-id/`, then put them in
[`scripts/common.sh`](../scripts/common.sh) (or set `FOLLOWER_PORT`, `LEADER_PORT`, `FRONT_CAM` and `WRIST_CAM` in the
environment). `lerobot-find-port` tells you which serial device is which arm.

Calibrate each arm once, with a fixed name (`--robot.id` / `--teleop.id`), so the calibration file is found again:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<follower_port> --robot.id=<follower_name>
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<leader_port>  --teleop.id=<leader_name>
```

`bash scripts/setup.sh` checks the conda env, LeRobot, the Hugging Face login, the GPU, serial and camera devices,
and whether both calibrations exist. `python scripts/leader_stream_test.py --port <leader_port> --id <leader_name>`
prints the leader's joint values live, without the follower.

## Lessons

- **Use MJPG for both cameras** (`fourcc: MJPG`). By default each sends uncompressed YUYV (about 150 Mbps), and with
  both cameras and both arm boards on one USB 2.0 hub, the front camera produced corrupted, jumping frames.
- **Wrist-roll offset between the arms.** `wrist_roll`'s zero point is wherever the joint happens to be during the
  "move to the middle" calibration step ([lerobot#3193](https://github.com/huggingface/lerobot/issues/3193)). Put both
  arms' wrists in the same physical pose (for example, jaws vertical) before pressing Enter on that step, for both arms.
- **A failing shoulder_lift motor.** `shoulder_lift` carries the whole arm. Ours started sticking and snapping
  (probably stripped gears) and was replaced. Get the right variant (STS3215 **12 V** for this follower; a 7.4 V one
  arrived first by mistake), and to set up just the new motor without walking all six, use
  `python scripts/setup_one_motor.py shoulder_lift` with only that motor connected.
- **Keep the Rerun viewer off while recording** (`--display_data=false`) if the recording loop freezes; it was the
  suspected cause here.
- **Don't press Ctrl+Z** during recording: it suspends the process while it still holds both arms and cameras.
