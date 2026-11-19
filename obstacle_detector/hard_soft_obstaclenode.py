#!/usr/bin/env python3
"""ROS 2 node that runs Ultralytics YOLO on camera images.

Input:
  sensor_msgs/msg/Image on ``camera_topic`` (default: /camerafeedback)

Output:
  vision_msgs/msg/Detection2DArray on ``detections_topic``

Each detection contains one class hypothesis and a pixel-space center/size
bounding box.  The message is intentionally standard so navigation can
subscribe without depending on a custom interface package.
"""

from __future__ import annotations

from typing import Any

import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class YoloObstacleDetector(Node):
    def __init__(self) -> None:
        super().__init__("yolo_obstacle_detector")

        self.declare_parameter("camera_topic", "/camerafeedback")
        self.declare_parameter("detections_topic", "/obstacles/detections")
        self.declare_parameter("model", "yolo26n.pt")
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "")
        self.declare_parameter("max_detections", 100)

        camera_topic = str(self.get_parameter("camera_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        model_path = str(self.get_parameter("model").value)

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
        except Exception as exc:
            self.get_logger().fatal(
                f"Could not load YOLO model '{model_path}': {exc}. "
                "Install ultralytics and make the model available before "
                "starting this node."
            )
            raise

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, camera_topic, self.image_callback, 10
        )
        self.publisher = self.create_publisher(
            Detection2DArray, detections_topic, 10
        )
        self.get_logger().info(
            f"Listening on {camera_topic}; publishing detections on "
            f"{detections_topic} using {model_path}"
        )

    def image_callback(self, image_msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert camera image: {exc}")
            return

        confidence = float(self.get_parameter("confidence").value)
        iou = float(self.get_parameter("iou").value)
        device = str(self.get_parameter("device").value)
        max_detections = int(self.get_parameter("max_detections").value)

        inference_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": confidence,
            "iou": iou,
            "verbose": False,
            "max_det": max_detections,
        }
        if device:
            inference_kwargs["device"] = device

        try:
            results = self.model.predict(**inference_kwargs)
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            return

        output = Detection2DArray()
        output.header = image_msg.header

        if not results:
            self.publisher.publish(output)
            return

        result = results[0]
        boxes = result.boxes
        names = result.names

        for index in range(len(boxes)):
            xywh = boxes.xywh[index].tolist()
            class_id = int(boxes.cls[index].item())
            score = float(boxes.conf[index].item())
            class_name = str(names[class_id])

            detection = Detection2D()
            detection.header = image_msg.header
            detection.bbox.center.position.x = float(xywh[0])
            detection.bbox.center.position.y = float(xywh[1])
            detection.bbox.size_x = float(xywh[2])
            detection.bbox.size_y = float(xywh[3])

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = score
            detection.results.append(hypothesis)
            output.detections.append(detection)

        self.publisher.publish(output)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
