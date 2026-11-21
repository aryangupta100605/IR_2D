#!/usr/bin/env python3
"""Display annotated ROS camera frames using local Ultralytics YOLO inference."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class DetectorVisualizerNode(Node):
    """Subscribe to a camera topic, draw YOLO boxes, and report detections."""

    def __init__(self) -> None:
        super().__init__("detector_visualizer_node")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("model", "yolo11n.pt")
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "")
        self.declare_parameter("window", "detector preview")
        self.declare_parameter("object_of_interest", "")
        self.declare_parameter("print_interval", 1.0)

        from ultralytics import YOLO

        model_path = str(self.get_parameter("model").value)
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        self.window = str(self.get_parameter("window").value)
        self.last_print_time = 0.0
        camera_topic = str(self.get_parameter("camera_topic").value)
        self.subscription = self.create_subscription(
            Image, camera_topic, self.image_callback, 10
        )
        self.get_logger().info(
            f"Showing local YOLO detections from {camera_topic} using {model_path}; "
            "press q or ESC to quit"
        )

    def image_callback(self, image_msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning(f"Could not convert camera frame: {error}")
            return

        counts = self.draw_yolo_detections(frame)
        self.print_object_counts(counts)
        self.draw_qr(frame)
        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.get_logger().info("Stopping detector preview")
            rclpy.shutdown()

    def draw_yolo_detections(self, frame: Any) -> Counter[str]:
        """Run local YOLO inference and draw every detection on the frame."""
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": float(self.get_parameter("confidence").value),
            "iou": float(self.get_parameter("iou").value),
            "max_det": 50,
            "verbose": False,
        }
        device = str(self.get_parameter("device").value)
        if device:
            kwargs["device"] = device

        try:
            results = self.model.predict(**kwargs)
        except Exception as error:
            self.get_logger().error(f"YOLO inference failed: {error}")
            return Counter()
        if not results or results[0].boxes is None:
            return Counter()

        result = results[0]
        counts: Counter[str] = Counter()
        for index in range(len(result.boxes)):
            class_id = int(result.boxes.cls[index].item())
            label = str(result.names[class_id]).strip().lower()
            confidence = float(result.boxes.conf[index].item())
            left, top, right, bottom = [
                int(value) for value in result.boxes.xyxy[index].tolist()
            ]
            counts[label] += 1
            color = (255, 0, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            self.draw_label(
                frame,
                f"{label}: {confidence:.2f}",
                left,
                max(top - 8, 20),
                color,
            )
        return counts

    def print_object_counts(self, counts: Counter[str]) -> None:
        """Print counts for the configured target, without flooding the terminal."""
        if not counts:
            return
        target_text = str(self.get_parameter("object_of_interest").value).strip().lower()
        targets = {item.strip() for item in target_text.split(",") if item.strip()}
        selected = counts if not targets else Counter(
            {label: count for label, count in counts.items() if label in targets}
        )
        if not selected:
            return

        now = time.monotonic()
        interval = max(0.0, float(self.get_parameter("print_interval").value))
        if now - self.last_print_time < interval:
            return
        self.last_print_time = now
        summary = ", ".join(
            f"{label}={count}" for label, count in sorted(selected.items())
        )
        self.get_logger().info(f"Detected object count: {summary}")

    def draw_qr(self, frame: Any) -> None:
        payload, points, _ = self.qr_detector.detectAndDecode(frame)
        if points is None:
            return
        corners = points[0].astype(int)
        for index in range(4):
            cv2.line(
                frame,
                tuple(corners[index]),
                tuple(corners[(index + 1) % 4]),
                (0, 255, 0),
                2,
            )
        label = "QR detected" if not payload else f"QR: {payload[:32]}"
        left = int(corners[:, 0].min())
        top = int(corners[:, 1].min())
        self.draw_label(frame, label, left, max(top - 8, 20), (0, 255, 0))

    @staticmethod
    def draw_label(
        frame: Any, text: str, x: int, y: int, color: tuple[int, int, int]
    ) -> None:
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def close(self) -> None:
        cv2.destroyAllWindows()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: DetectorVisualizerNode | None = None
    try:
        node = DetectorVisualizerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

