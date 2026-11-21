# YOLO obstacle detector for ROS 2

This package subscribes to camera frames, runs Ultralytics YOLO, and publishes
standard `vision_msgs/msg/Detection2DArray` messages for navigation.

## Build and run

From the root of a ROS 2 workspace:

```bash
cd src
git clone <this-package-location> yolo_obstacle_detector
cd ..
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -r src/yolo_obstacle_detector/requirements.txt
colcon build --packages-select yolo_obstacle_detector
source install/setup.bash
ros2 run yolo_obstacle_detector detector_node
```

The default topics are:

- Input: `/camera/color/image_raw` (`sensor_msgs/msg/Image`)
- Output: `/obstacles/detections` (`vision_msgs/msg/Detection2DArray`)

The default model is `yolo26n.pt`. Ultralytics downloads it on first use when
the machine has network access, or it can be replaced with a local path.

## Parameters

```bash
ros2 run yolo_obstacle_detector detector_node --ros-args \
  -p camera_topic:=/camera/color/image_raw \
  -p model:=/models/yolo26n.pt \
  -p confidence:=0.40 \
  -p device:=cuda:0
```

The bounding box fields are pixel coordinates relative to the source image:
`center.position.x/y`, `size_x`, and `size_y`. Each detection's class and
confidence are in `results[0].hypothesis.class_id` and `.score`.

## Train the hazard-symbol model

[`hazard_training/train_hazard_yolo.py`](hazard_training/train_hazard_yolo.py)
maps the ISO 7010 source signs to the five requested model classes:

| Model class | Source signs |
| --- | --- |
| `high_voltage` | W012 electricity hazard |
| `radioactive` | W003 radioactive material |
| `chemical` | W016 toxic material, W023 corrosive substance |
| `biological` | W009 biological hazard |
| `flammable` | W021 flammable material |

Generate a synthetic object-detection dataset and train it with:

```bash
python hazard_training/train_hazard_yolo.py all --samples-per-sign 250 --epochs 80
```

The dataset is written to `hazard_training/dataset`; the trained weights are
written below `hazard_training/dataset/runs/hazard_yolo/weights/`. Use
`prepare` or `train` separately when needed. The script generates varied
industrial backgrounds, perspective, lighting, blur, noise, and partial
occlusion while keeping the original pictograms unchanged.

## Overseer-controlled hazard inspection

The runtime flow is split into three nodes:

1. `overseer_node` listens for a non-empty `std_msgs/msg/String` navigation
   trigger on `/navigation`, commands locomotion with JSON on `/locomotion`,
   and publishes each collected view on `/hazard/views`.
2. `hazard_detector_node` runs continuously in the background, checks each
   view with the trained model, and publishes JSON notifications on
   `/hazard/notifications`.
3. `qr_detector_node` runs continuously in the background, checks each view
   for QR presence, and publishes JSON notifications on `/qr/notifications`.
4. The overseer runs the COCO `yolo11n.pt` person detector on each sweep view.
   Only views containing a person are forwarded to the QR detector; hazard
   detection still receives every sweep view.
5. The overseer adds each new label to its priority list and sends locomotion
   an `approach_label` command containing the label and pixel bounding box.

Start the two coordination nodes in separate processes after training:

```bash
ros2 run yolo_obstacle_detector overseer_node
ros2 run yolo_obstacle_detector hazard_detector_node --ros-args \
  -p model:=weights/best.pt
ros2 run yolo_obstacle_detector qr_detector_node
```

Locomotion should handle the JSON commands `collect_views`, `turn_relative`,
and `approach_label`. The existing `/obstacles/detections` subscription is
kept for compatibility with the generic detector, but hazard control now
uses the dedicated notification topic. QR presence uses `/qr/views` and
`/qr/notifications`; QR payload decryption remains in `qr_query_node`.

The overseer uses `yolo11n.pt` for human detection by default. Override it or
tune the gate with ROS parameters:

```bash
ros2 run yolo_obstacle_detector overseer_node --ros-args \
  -p human_model:=/models/yolo11n.pt \
  -p human_confidence:=0.45 \
  -p human_device:=cpu
```

## ROS 2 local YOLO camera visualizer test

This test runs the local `detector_visualizer_node` only. It does not query GPT
or any perception server. It subscribes to `sensor_msgs/msg/Image`, opens an
annotated OpenCV window, draws a box around each YOLO detection, and prints
object counts in the terminal.

On a machine with ROS 2 installed, download the repository into a ROS 2
workspace and build it:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone --branch integration https://github.com/aryangupta100605/IR_2D.git yolo_obstacle_detector
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -r src/yolo_obstacle_detector/requirements.txt
colcon build --symlink-install --packages-select yolo_obstacle_detector
source install/setup.bash
```

The node expects the dog's camera images on `/camera/color/image_raw`. Start the camera or
another ROS 2 node that publishes that topic, then run the visualizer in a new
sourced terminal:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run yolo_obstacle_detector detector_visualizer_node
```

To print only a particular object count, such as the number of people, use
`object_of_interest`. Leave it empty to print counts for every detected class:

```bash
ros2 run yolo_obstacle_detector detector_visualizer_node --ros-args \
  -p camera_topic:=/camera/color/image_raw \
  -p model:=/home/$USER/ros2_ws/src/yolo_obstacle_detector/yolo11n.pt \
  -p object_of_interest:=person \
  -p confidence:=0.35
```

If the camera publishes another topic, find image topics with
`ros2 topic list` and replace `/camera/color/image_raw` in the command. Press `q` or
`Esc` in the preview window to stop the test.

## Webcam test

Before connecting the robot camera, publish a local webcam on the default
camera topic:

```bash
ros2 run yolo_obstacle_detector webcam_test_node
```

The default webcam device is `0`. For testing against the dog's camera, use
`/camera/color/image_raw` as the topic.
For another webcam or topic:

```bash
ros2 run yolo_obstacle_detector webcam_test_node --ros-args \
  -p device:=1 -p topic:=/camera/color/image_raw -p fps:=10
```

Run the webcam publisher and detector nodes in separate sourced terminals.
To trigger the coordinator's room scan:

```bash
ros2 topic pub --once /navigation std_msgs/msg/String "{data: test_scan}"
```

For a direct visual detector test using the webcam, publish the webcam on
`/camera/color/image_raw` and run this node in separate sourced terminals:

```bash
ros2 run yolo_obstacle_detector webcam_test_node --ros-args \
  -p topic:=/camera/color/image_raw
ros2 run yolo_obstacle_detector detector_visualizer_node --ros-args \
  -p camera_topic:=/camera/color/image_raw \
  -p object_of_interest:=person
```

YOLO boxes are drawn in blue and QR boxes are drawn in green. The terminal
prints messages such as `Detected object count: person=2`. Press `q` or `Esc`
in the preview window to stop the visualizer.

ROS 2 does not require a central server. The processes only need the same ROS
2 installation, sourced environment, and `ROS_DOMAIN_ID`.

## Windows test without ROS 2

The removable `windows_detector_test` entry point runs the human, hazard, and
QR detectors directly from a laptop webcam, image, or video. Human detection
uses `yolo11n.pt`; QR detection is triggered only on frames containing a
person. Hazard detection uses `hazard_detection/weights/best.pt` by default:

```powershell
python -m pip install -r requirements.txt
python -m obstacle_detector.windows_detector_test --source 0
```

Use `--source .\photo.jpg` for a single image, `--model path\to\best.pt` for
different hazard weights, or `--no-display --json --frames 1` for a headless
check. Human boxes are blue, hazard boxes are red, and QR boxes are green.
Press `q` or `Esc` to stop the annotated preview. The same tool is available
after package installation as `windows_detector_test`.

## GPT webcam test

The GPT query node defaults to the LAN perception server at
`http://192.168.10.141:8000`. For a ROS-free webcam test, run:

```powershell
.\.venv\Scripts\python.exe -m obstacle_detector.gpt_query_node --webcam-test
```

The preview uses the default prompt to classify the scene as `kitchen`,
`market`, `workshop`, `clinic`, or `cleaning_room`, while also requesting human,
QR, and hazard fields. After the GPT response, the local person and QR detectors
run on the same frame and print their results. Press `Space` to send the current
frame to the server's `/analyze` endpoint. Press
`q` or `Esc` to exit. The default full group ID is `group03_6769`; override it with
`--group` if needed:

```powershell
.\.venv\Scripts\python.exe -m obstacle_detector.gpt_query_node `
  --webcam-test --group group03_6769
```

Override the server or prompt with `--server` and `--prompt` if needed.

## Single-view mission flow

The overseer defaults to one full-view capture after a non-empty navigation
trigger. It does not rotate the robot for the default `scan_angles_degrees:=0`
setting. The image is sent once to GPT, and the resulting mission report uses
the fields `group_id`, `task_id`, `task_type`, `scene_type`, `label`, `pos`, and
`message`. Override the report identity when needed:

```bash
ros2 run yolo_obstacle_detector overseer_node --ros-args \
  -p report_group_id:=group03_6769 \
  -p report_task_id:=M01 \
  -p report_task_type:=object
```
