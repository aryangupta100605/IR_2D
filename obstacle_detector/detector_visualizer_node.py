#!/usr/bin/env python3
"""Display webcam frames with hazard-model and OpenCV QR bounding boxes."""

from __future__ import annotations

from typing import Any

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


HAZARD_CLASSES = {
    "high_voltage",
    "radioactive",
    "chemical",
    "biological",
    "flammable",
}
PERSON_CLASS = "person"


class DetectorVisualizerNode(Node):
    """Run both detectors on webcam ROS images and show their boxes."""

    def __init__(self) -> None:
        super().__init__("detector_visualizer_node")
        self.declare_parameter("camera_topic", "/camerafeedback")
        self.declare_parameter(
            "model", "hazard_training/dataset/runs/hazard_yolo/weights/best.pt"
        )
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "")
        self.declare_parameter("window", "detector preview")

        from ultralytics import YOLO

        model_path = str(self.get_parameter("model").value)
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        self.window = str(self.get_parameter("window").value)
        camera_topic = str(self.get_parameter("camera_topic").value)
        self.subscription = self.create_subscription(
            Image, camera_topic, self.image_callback, 10
        )
        self.get_logger().info(
            f"Showing person, hazard, and QR detections from {camera_topic} "
            f"using {model_path}; press q or ESC to quit"
        )

    def image_callback(self, image_msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning("Could not convert camera frame: %s", error)
            return

        self.draw_hazards(frame)
        self.draw_qr(frame)
        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.get_logger().info("Stopping detector preview")
            rclpy.shutdown()

    def draw_hazards(self, frame: Any) -> None:
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": float(self.get_parameter("confidence").value),
            "iou": float(self.get_parameter("iou").value),
            "max_det": 20,
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
            if label not in HAZARD_CLASSES and label != PERSON_CLASS:
                continue
            confidence = float(result.boxes.conf[index].item())
            left, top, right, bottom = [
                int(value) for value in result.boxes.xyxy[index].tolist()
            ]
            color = (255, 0, 0) if label == PERSON_CLASS else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            self.draw_label(
                frame,
                f"{label.upper()}: {confidence:.2f}",
                left,
                max(top - 8, 20),
                color,
            )

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
