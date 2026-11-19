#!/usr/bin/env python3
"""Run the trained hazard-sign model as an independent background ROS node.

The overseer publishes selected scan views on ``views_topic``.  This node
checks every view and publishes one JSON notification per detected label on
``notification_topic``.  Locomotion details stay in the overseer, so this
node remains a detector rather than a controller.
"""

from __future__ import annotations

import json
from typing import Any

import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


HAZARD_CLASSES = {
    "high_voltage",
    "radioactive",
    "chemical",
    "biological",
    "flammable",
}


class HazardDetectorNode(Node):
    """Infer hazard labels from views supplied by the overseer."""

    def __init__(self) -> None:
        super().__init__("hazard_detector_node")
        self.declare_parameter("views_topic", "/hazard/views")
        self.declare_parameter("notification_topic", "/hazard/notifications")
        self.declare_parameter(
            "model", "hazard_training/dataset/runs/hazard_yolo/weights/best.pt"
        )
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "")
        self.declare_parameter("max_detections", 20)

        from ultralytics import YOLO

        model_path = str(self.get_parameter("model").value)
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        views_topic = str(self.get_parameter("views_topic").value)
        notification_topic = str(self.get_parameter("notification_topic").value)
        self.subscription = self.create_subscription(
            Image, views_topic, self.image_callback, 10
        )
        self.publisher = self.create_publisher(String, notification_topic, 10)
        self.get_logger().info(
            "Checking overseer views on %s with %s; notifications on %s",
            views_topic,
            model_path,
            notification_topic,
        )

    def image_callback(self, image_msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning("Could not convert hazard view: %s", error)
            return

        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": float(self.get_parameter("confidence").value),
            "iou": float(self.get_parameter("iou").value),
            "max_det": int(self.get_parameter("max_detections").value),
            "verbose": False,
        }
        device = str(self.get_parameter("device").value)
        if device:
            kwargs["device"] = device
        try:
            results = self.model.predict(**kwargs)
        except Exception as error:
            self.get_logger().error("Hazard inference failed: %s", error)
            return
        if not results:
            return

        result = results[0]
        for index in range(len(result.boxes)):
            class_id = int(result.boxes.cls[index].item())
            label = str(result.names[class_id]).strip().lower()
            if label not in HAZARD_CLASSES:
                continue
            xywh = [float(value) for value in result.boxes.xywh[index].tolist()]
            self.publish_notification(image_msg, label, float(result.boxes.conf[index].item()), xywh)

    def publish_notification(
        self, image_msg: Image, label: str, confidence: float, xywh: list[float]
    ) -> None:
        scan_id, view_angle = self.parse_view_context(image_msg.header.frame_id)
        notification = {
            "event": "hazard_detected",
            "scan_id": scan_id,
            "view_angle_degrees": view_angle,
            "label": label,
            "confidence": confidence,
            "bbox": {"center_x": xywh[0], "center_y": xywh[1], "width": xywh[2], "height": xywh[3]},
            "camera_stamp": {
                "sec": int(image_msg.header.stamp.sec),
                "nanosec": int(image_msg.header.stamp.nanosec),
            },
        }
        message = String()
        message.data = json.dumps(notification)
        self.publisher.publish(message)

    @staticmethod
    def parse_view_context(frame_id: str) -> tuple[str, float | None]:
        parts = frame_id.split("|", 1)
        scan_id = parts[0] or "unknown"
        try:
            return scan_id, float(parts[1]) if len(parts) == 2 else None
        except ValueError:
            return scan_id, None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HazardDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
