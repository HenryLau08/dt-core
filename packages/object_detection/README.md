# object_detection

ONNX-based duckie object detection ROS node for `dt-core`.

## Package layout

```
dt-core/
├── packages/
│   └── object_detection/
│       ├── assets/
│       │   └── best.onnx          ← place your trained model here
│       ├── config/
│       │   └── object_detection_node/
│       │       └── default.yaml   ← conf_threshold, stop_distance, forward_pwm
│       ├── launch/
│       │   └── object_detection_node.launch
│       ├── src/
│       │   ├── object_detection_node.py   ← ROS node entry point
│       │   └── solution/
│       │       ├── __init__.py
│       │       ├── config.py      ← paths and constants (MODEL_PATH → assets/)
│       │       ├── model.py       ← MLModel: stop behaviour
│       │       └── model_afwijk.py ← MLModel: avoidance state machine
│       ├── CMakeLists.txt
│       ├── package.xml
│       └── setup.py
```

## Before launching

1. **Copy your trained model** into the package:
   ```
   cp /path/to/best.onnx dt-core/packages/object_detection/assets/best.onnx
   ```
   Optionally also copy `classes.yaml` to the same folder.

2. **Add `onnxruntime` to dt-core dependencies** if not already present:
   ```
   # dt-core/dependencies-py3.txt
   onnxruntime>=1.17.0
   ```

3. **Rebuild the Docker image**:
   ```
   dts devel build -f -H ROBOTNAME.local
   ```

## Launching standalone

```bash
roslaunch object_detection object_detection_node.launch veh:=ROBOTNAME
```

## Integrating into lane following

Add the following `<include>` to
`dt-core/packages/duckietown_demos/launch/master.launch`
(alongside the existing obstacle_detection include):

```xml
<include file="$(find object_detection)/launch/object_detection_node.launch">
    <arg name="veh" value="$(arg veh)"/>
</include>
```

Then remap the `~wheels_cmd` output of this node into whatever topic your
lane controller is listening on, or wire `~object_detected` into the FSM /
obstacle_detection node to trigger a virtual stop line.

## Topic remapping reference

| Direction  | Topic (relative)          | Message type                    |
|------------|---------------------------|---------------------------------|
| Subscribe  | `~image/compressed`       | `sensor_msgs/CompressedImage`   |
| Subscribe  | `~camera_info`            | `sensor_msgs/CameraInfo`        |
| Publish    | `~wheels_cmd`             | `duckietown_msgs/WheelsCmdStamped` |
| Publish    | `~object_detected`        | `duckietown_msgs/BoolStamped`   |
