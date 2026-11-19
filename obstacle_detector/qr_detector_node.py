#!/usr/bin/env python3
"""Detect QR presence in views collected by the overseer.

This node reports presence and location only. Decryption and task reporting
remain responsibilities of the existing ``qr_query_node``.
"""

from __future__ import annotations

import json

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class QrDetectorNode(Node):
    """Run QR detection continuously on overseer-selected views."""

    def __init__(self) -> None:
        super().__init__("qr_detector_node")
        self.declare_parameter("views_topic", "/qr/views")
        self.declare_parameter("notification_topic", "/qr/notifications")
        views_topic = str(self.get_parameter("views_topic").value)
        notification_topic = str(self.get_parameter("notification_topic").value)

        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        self.subscription = self.create_subscription(
            Image, views_topic, self.image_callback, 10
        )
        self.publisher = self.create_publisher(String, notification_topic, 10)
        self.get_logger().info(
            "Checking collected QR views on %s; notifications on %s",
            views_topic,
            notification_topic,
        )

    def image_callback(self, image_msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning("Could not convert QR view: %s", error)
            return

        payload, points, _ = self.qr_detector.detectAndDecode(frame)
        if points is None:
            return
        scan_id, angle = self.parse_view_context(image_msg.header.frame_id)
        notification = {
            "event": "qr_detected",
            "detected": True,
            "scan_id": scan_id,
            "view_angle_degrees": angle,
            "payload_present": bool(payload),
            "bbox": self.bounding_box(points),
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
        try:
            angle = float(parts[1]) if len(parts) == 2 else None
        except ValueError:
            angle = None
        return parts[0] or "unknown", angle

    @staticmethod
    def bounding_box(points: object) -> dict[str, float] | None:
        try:
            corners = points[0]  # type: ignore[index]
            xs = [float(point[0]) for point in corners]
            ys = [float(point[1]) for point in corners]
        except (IndexError, TypeError):
            return None
        if len(xs) != 4 or len(ys) != 4:
            return None
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        return {
            "center_x": (left + right) / 2.0,
            "center_y": (top + bottom) / 2.0,
            "width": right - left,
            "height": bottom - top,
        }


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = QrDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
