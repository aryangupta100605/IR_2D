#!/usr/bin/env python3
"""Publish a local webcam as a ROS 2 sensor_msgs/Image stream."""

from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class WebcamTestNode(Node):
    """Read frames from a local webcam and publish them as ROS images."""

    def __init__(self) -> None:
        super().__init__("webcam_test_node")
        self.declare_parameter("device", 0)
        self.declare_parameter("topic", "/camerafeedback")
        self.declare_parameter("frame_id", "webcam")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 15.0)

        device = int(self.get_parameter("device").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)
        topic = str(self.get_parameter("topic").value)

        self.capture = cv2.VideoCapture(device)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam device {device}")

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, topic, 10)
        self.timer = self.create_timer(1.0 / max(fps, 1.0), self.publish_frame)
        self.get_logger().info(
            "Publishing webcam %d on %s at approximately %.1f FPS",
            device,
            topic,
            fps,
        )

    def publish_frame(self) -> None:
        success, frame = self.capture.read()
        if not success:
            self.get_logger().warning("Could not read a frame from the webcam")
            return

        message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        self.publisher.publish(message)

    def close(self) -> None:
        self.capture.release()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: WebcamTestNode | None = None
    try:
        node = WebcamTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
