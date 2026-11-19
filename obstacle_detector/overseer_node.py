#!/usr/bin/env python3
"""Coordinate navigation-triggered views and hazard inspection.

This is intentionally a framework around provisional ROS interfaces:

* navigation publishes a ``std_msgs/String`` trigger on ``navigation_topic``;
* the node commands view collection and relative turns as JSON on
  ``locomotion_topic``;
* captured views are forwarded to the standalone hazard detector;
* the COCO YOLO model passively checks each captured view for people;
* QR views are forwarded only when a person is present in that view;
* GPT Vision returns strict JSON describing QR codes and hazard symbols;
* a closer approach is requested when either is found.

Replace the JSON String topics with project-specific ROS messages once the
navigation and locomotion interfaces are finalized.  The coordinator does not
decrypt QR payloads; it only requests a closer view.  The existing QR node can
then decode the payload from the closer camera view.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


HAZARD_CLASSES = {
    "high_voltage",
    "radioactive",
    "chemical",
    "biological",
    "flammable",
}
PERSON_CLASS = "person"


class SceneUnderstandingNode(Node):
    """Run a turn-and-look scan after navigation asks for room understanding."""

    def __init__(self) -> None:
        super().__init__("scene_understanding_node")

        self.declare_parameter("camera_topic", "/camerafeedback")
        self.declare_parameter("navigation_topic", "/navigation")
        self.declare_parameter("locomotion_topic", "/locomotion")
        self.declare_parameter("detections_topic", "/obstacles/detections")
        self.declare_parameter("hazard_views_topic", "/hazard/views")
        self.declare_parameter("hazard_notification_topic", "/hazard/notifications")
        self.declare_parameter("qr_views_topic", "/qr/views")
        self.declare_parameter("qr_notification_topic", "/qr/notifications")
        self.declare_parameter("object_notification_topic", "/object/notifications")
        self.declare_parameter("navigation_coordinates_topic", "/navigation/coordinates")
        self.declare_parameter("scene_result_topic", "/scene/understanding")
        self.declare_parameter("scan_angles_degrees", "0")
        self.declare_parameter("turn_settle_seconds", 1.5)
        self.declare_parameter("approach_distance_meters", 0.75)
        self.declare_parameter("request_timeout_seconds", 30.0)
        self.declare_parameter("gpt_query_topic", "/gpt/query")
        self.declare_parameter("gpt_result_topic", "/gpt/results")
        self.declare_parameter("report_group_id", "group03_6769")
        self.declare_parameter("report_task_id", "M01")
        self.declare_parameter("report_task_type", "object")
        self.declare_parameter("human_model", "yolo11n.pt")
        self.declare_parameter("human_confidence", 0.35)
        self.declare_parameter("human_iou", 0.45)
        self.declare_parameter("human_device", "")
        self.declare_parameter("human_max_detections", 20)

        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        from ultralytics import YOLO

        human_model_path = str(self.get_parameter("human_model").value)
        self.human_model = YOLO(human_model_path)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_header_stamp: dict[str, int] = {"sec": 0, "nanosec": 0}
        self.latest_yolo_hazards: set[str] = set()
        self.priority_list: list[dict[str, Any]] = []
        self.priority_keys: set[tuple[str, str]] = set()
        self.scan_active = False
        self.scan_id: str | None = None
        self.scan_angles: list[float] = []
        self.scan_index = 0
        self.next_capture_time = 0.0
        self.approach_requested = False
        self.pending_object_reports: dict[tuple[str, str], dict[str, Any]] = {}
        self.object_coordinates: dict[tuple[str, str], list[float]] = {}

        camera_topic = str(self.get_parameter("camera_topic").value)
        navigation_topic = str(self.get_parameter("navigation_topic").value)
        locomotion_topic = str(self.get_parameter("locomotion_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        hazard_views_topic = str(self.get_parameter("hazard_views_topic").value)
        hazard_notification_topic = str(
            self.get_parameter("hazard_notification_topic").value
        )
        qr_views_topic = str(self.get_parameter("qr_views_topic").value)
        qr_notification_topic = str(self.get_parameter("qr_notification_topic").value)
        object_notification_topic = str(
            self.get_parameter("object_notification_topic").value
        )
        navigation_coordinates_topic = str(
            self.get_parameter("navigation_coordinates_topic").value
        )
        result_topic = str(self.get_parameter("scene_result_topic").value)
        gpt_query_topic = str(self.get_parameter("gpt_query_topic").value)
        gpt_result_topic = str(self.get_parameter("gpt_result_topic").value)

        self.camera_subscription = self.create_subscription(
            Image, camera_topic, self.camera_callback, 10
        )
        self.navigation_subscription = self.create_subscription(
            String, navigation_topic, self.navigation_callback, 10
        )
        self.detection_subscription = self.create_subscription(
            Detection2DArray, detections_topic, self.detection_callback, 10
        )
        self.hazard_subscription = self.create_subscription(
            String, hazard_notification_topic, self.hazard_notification_callback, 10
        )
        self.qr_subscription = self.create_subscription(
            String, qr_notification_topic, self.qr_notification_callback, 10
        )
        self.object_subscription = self.create_subscription(
            String, object_notification_topic, self.object_notification_callback, 10
        )
        self.coordinates_subscription = self.create_subscription(
            String,
            navigation_coordinates_topic,
            self.coordinates_callback,
            10,
        )
        self.gpt_result_subscription = self.create_subscription(
            String, gpt_result_topic, self.gpt_result_callback, 10
        )
        self.locomotion_publisher = self.create_publisher(String, locomotion_topic, 10)
        self.hazard_view_publisher = self.create_publisher(Image, hazard_views_topic, 10)
        self.qr_view_publisher = self.create_publisher(Image, qr_views_topic, 10)
        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.gpt_query_publisher = self.create_publisher(String, gpt_query_topic, 10)
        self.scan_timer = self.create_timer(0.1, self.scan_tick)

        self.get_logger().info(
            "Overseer ready: trigger=%s camera=%s locomotion=%s hazard_views=%s qr_views=%s human_model=%s",
            navigation_topic,
            camera_topic,
            locomotion_topic,
            hazard_views_topic,
            qr_views_topic,
            human_model_path,
        )

    def camera_callback(self, message: Image) -> None:
        """Keep only the most recent JPEG; scan_tick chooses when to use it."""
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning("Could not convert camera frame: %s", error)
            return
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return
        with self.lock:
            self.latest_jpeg = encoded.tobytes()
            self.latest_header_stamp = {
                "sec": int(message.header.stamp.sec),
                "nanosec": int(message.header.stamp.nanosec),
            }

    def detection_callback(self, message: Detection2DArray) -> None:
        """Cache hazard classes found by the local YOLO detector, if running."""
        found: set[str] = set()
        for detection in message.detections:
            for result in detection.results:
                name = str(result.hypothesis.class_id).strip().lower()
                if name in HAZARD_CLASSES:
                    found.add(name)
        with self.lock:
            self.latest_yolo_hazards = found

    def navigation_callback(self, message: String) -> None:
        """Start a scan for any non-empty navigation trigger."""
        if not message.data.strip():
            return
        if self.scan_active:
            self.get_logger().info("Ignoring navigation trigger while scan is active")
            return
        self.start_scan(message.data)

    def hazard_notification_callback(self, message: String) -> None:
        """Add a detector observation to the priority list and approach it."""
        try:
            notification = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring invalid hazard notification: %s", message.data)
            return

        label = str(notification.get("label", "")).strip().lower()
        scan_id = str(notification.get("scan_id", self.scan_id or ""))
        if label not in HAZARD_CLASSES or not scan_id:
            return
        key = (scan_id, label)
        if key in self.priority_keys:
            return

        priority = {**notification, "label": label, "scan_id": scan_id}
        self.priority_keys.add(key)
        self.priority_list.append(priority)
        self.get_logger().warning("Hazard priority added: %s", json.dumps(priority))
        self.publish_json(self.result_publisher, {
            "event": "hazard_priority_added",
            "scan_id": scan_id,
            "priority_list": self.priority_list,
        })
        self.move_closer_to_label(priority)

    def qr_notification_callback(self, message: String) -> None:
        """Add a QR observation to the priority list and approach it."""
        try:
            notification = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring invalid QR notification: %s", message.data)
            return
        if not notification.get("detected"):
            return
        scan_id = str(notification.get("scan_id", self.scan_id or ""))
        if not scan_id:
            return
        key = (scan_id, "qr")
        if key in self.priority_keys:
            return

        priority = {**notification, "scan_id": scan_id, "label": "qr"}
        self.priority_keys.add(key)
        self.priority_list.append(priority)
        self.get_logger().info("QR priority added: %s", json.dumps(priority))
        if notification.get("request_coordinates"):
            self.publish_json(self.locomotion_publisher, {
                "command": "request_coordinates",
                "scan_id": scan_id,
                "pos": notification.get("pos", [2.04, 3.06]),
                "reason": "qr_scanned",
            })
        self.publish_json(self.result_publisher, {
            "event": "qr_priority_added",
            "scan_id": scan_id,
            "priority_list": self.priority_list,
        })
        self.move_closer_to_label(priority)

    def gpt_result_callback(self, message: String) -> None:
        """Forward a completed GPT query through the normal scene-result path."""
        try:
            payload = json.loads(message.data)
            context = payload["context"]
            result = payload["result"]
            if not isinstance(context, dict) or not isinstance(result, dict):
                raise ValueError("context and result must be objects")
            if isinstance(result.get("result"), dict):
                result = result["result"]
            elif isinstance(result.get("result"), str):
                result = json.loads(result["result"])
            elif isinstance(result.get("analysis"), dict):
                result = result["analysis"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning("Ignoring invalid GPT result: %s", error)
            return
        self.publish_result(context, result)

    def object_notification_callback(self, message: String) -> None:
        """Ask navigation for coordinates for every object GPT classified."""
        try:
            notification = json.loads(message.data)
            scan_id = str(notification.get("scan_id", self.scan_id or ""))
            labels = notification.get("labels", [])
            if not scan_id or not isinstance(labels, list):
                raise ValueError("object notification needs scan_id and labels")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning("Ignoring invalid object notification: %s", error)
            return

        clean_labels = sorted({str(label).strip().lower() for label in labels if str(label).strip()})
        if not clean_labels:
            return
        self.publish_json(self.locomotion_publisher, {
            "command": "request_coordinates",
            "scan_id": scan_id,
            "labels": clean_labels,
            "reason": "objects_classified",
        })

    def coordinates_callback(self, message: String) -> None:
        """Complete pending object reports from a navigation coordinate response.

        Expected response JSON: {"scan_id": "...", "label": "food",
        "pos": [x, y]}.
        """
        try:
            response = json.loads(message.data)
            scan_id = str(response["scan_id"])
            label = str(response["label"]).strip().lower()
            pos = response["pos"]
            if (
                not isinstance(pos, list)
                or len(pos) != 2
                or not all(isinstance(value, (int, float)) for value in pos)
            ):
                raise ValueError("pos must contain two numbers")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning("Ignoring invalid navigation coordinates: %s", error)
            return

        key = (scan_id, label)
        self.object_coordinates[key] = [float(pos[0]), float(pos[1])]
        self.publish_pending_object_report(key)

    def publish_pending_object_report(self, key: tuple[str, str]) -> None:
        """Publish one object report once its navigation position is available."""
        report = self.pending_object_reports.get(key)
        position = self.object_coordinates.get(key)
        if report is None or position is None:
            return
        self.pending_object_reports.pop(key)
        self.object_coordinates.pop(key)
        report = {**report, "pos": position}
        self.get_logger().info("Mission report: %s", json.dumps(report, ensure_ascii=False))
        self.publish_json(self.result_publisher, report)

    def move_closer_to_label(self, priority: dict[str, Any]) -> None:
        """Ask locomotion to center on and approach the detected hazard label."""
        self.publish_json(self.locomotion_publisher, {
            "command": "approach_label",
            "scan_id": priority["scan_id"],
            "label": priority["label"],
            "bbox": priority.get("bbox"),
            "distance_meters": float(
                self.get_parameter("approach_distance_meters").value
            ),
            "reason": "hazard_detector_notification",
        })

    def start_scan(self, trigger: str) -> None:
        raw_angles = str(self.get_parameter("scan_angles_degrees").value)
        try:
            angles = [float(value.strip()) for value in raw_angles.split(",") if value.strip()]
        except ValueError:
            self.get_logger().error("Invalid scan_angles_degrees value: %s", raw_angles)
            return
        if not angles:
            self.get_logger().error("At least one scan angle is required")
            return

        self.scan_id = uuid.uuid4().hex
        self.scan_angles = angles
        self.scan_index = 0
        self.scan_active = True
        self.approach_requested = False
        self.priority_list = []
        self.priority_keys = set()
        self.get_logger().info("Starting room scan %s after trigger: %s", self.scan_id, trigger)
        self.publish_json(self.locomotion_publisher, {
            "command": "collect_views",
            "scan_id": self.scan_id,
            "angles_degrees": angles,
            "settle_seconds": float(self.get_parameter("turn_settle_seconds").value),
        })
        if len(angles) > 1:
            self.publish_turn(angles[0])
        else:
            # A single angle means the current camera view is already the full
            # mission view; do not rotate the robot before capturing it.
            self.next_capture_time = time.monotonic() + float(
                self.get_parameter("turn_settle_seconds").value
            )

    def publish_turn(self, angle_degrees: float) -> None:
        command = {
            "command": "turn_relative",
            "scan_id": self.scan_id,
            "angle_degrees": angle_degrees,
            "settle_seconds": float(self.get_parameter("turn_settle_seconds").value),
        }
        self.publish_json(self.locomotion_publisher, command)
        self.next_capture_time = time.monotonic() + float(
            self.get_parameter("turn_settle_seconds").value
        )

    def scan_tick(self) -> None:
        """After each turn, submit the latest view and advance the sweep."""
        if not self.scan_active or time.monotonic() < self.next_capture_time:
            return
        with self.lock:
            image = self.latest_jpeg
            stamp = dict(self.latest_header_stamp)
            yolo_hazards = sorted(self.latest_yolo_hazards)
        if image is None:
            self.get_logger().warning("No camera frame available for scan view %d", self.scan_index)
        else:
            human_detections = self.detect_humans(image)
            self.publish_hazard_view(
                image,
                stamp,
                self.scan_id or "unknown",
                self.scan_angles[self.scan_index],
                publish_qr=bool(human_detections),
            )
            self.submit_view(
                image,
                stamp,
                yolo_hazards,
                self.scan_angles[self.scan_index],
                human_detections,
            )

        self.scan_index += 1
        if self.scan_index >= len(self.scan_angles):
            self.scan_active = False
            self.get_logger().info("Room scan %s complete", self.scan_id)
            return
        self.publish_turn(self.scan_angles[self.scan_index])

    def submit_view(
        self,
        image: bytes,
        stamp: dict[str, int],
        yolo_hazards: list[str],
        angle: float,
        human_detections: list[dict[str, Any]],
    ) -> None:
        """Submit a view to GPT, gating local QR inspection on human presence."""
        qr_payload = self.decode_qr(image) if human_detections else None
        context = {
            "scan_id": self.scan_id,
            "view_angle_degrees": angle,
            "camera_stamp": stamp,
            "local_yolo_hazards": yolo_hazards,
            "human_present": bool(human_detections),
            "human_detections": human_detections,
            "qr_triggered": bool(human_detections),
            "local_qr_detected": qr_payload is not None,
        }
        self.executor.submit(self.trigger_gpt_query, image, context)

    def detect_humans(self, jpeg: bytes) -> list[dict[str, Any]]:
        """Detect people in one captured view using the COCO YOLO model."""
        array = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return []

        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": float(self.get_parameter("human_confidence").value),
            "iou": float(self.get_parameter("human_iou").value),
            "max_det": int(self.get_parameter("human_max_detections").value),
            "verbose": False,
        }
        device = str(self.get_parameter("human_device").value)
        if device:
            kwargs["device"] = device
        try:
            results = self.human_model.predict(**kwargs)
        except Exception as error:
            self.get_logger().error("Human inference failed: %s", error)
            return []
        if not results:
            return []

        result = results[0]
        detections: list[dict[str, Any]] = []
        for index in range(len(result.boxes)):
            class_id = int(result.boxes.cls[index].item())
            label = str(result.names[class_id]).strip().lower()
            if label != PERSON_CLASS:
                continue
            left, top, right, bottom = (
                float(value) for value in result.boxes.xyxy[index].tolist()
            )
            detections.append(
                {
                    "label": label,
                    "confidence": float(result.boxes.conf[index].item()),
                    "bbox": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    },
                }
            )
        if detections:
            self.get_logger().info(
                "Human detected at scan angle %.1f: %d person(s)",
                self.scan_angles[self.scan_index],
                len(detections),
            )
        return detections

    def decode_qr(self, jpeg: bytes) -> str | None:
        array = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        payload, _, _ = self.qr_detector.detectAndDecode(frame)
        return payload or None

    def trigger_gpt_query(self, image: bytes, context: dict[str, Any]) -> None:
        """Tell the dedicated GPT node to query the LAN perception server."""
        self.publish_json(self.gpt_query_publisher, {
            "image_base64": base64.b64encode(image).decode("ascii"),
            "context": context,
        })

    def publish_result(self, context: dict[str, Any], gpt_result: dict[str, Any]) -> None:
        hazards = sorted(
            set(gpt_result.get("hazards", [])) | set(context["local_yolo_hazards"])
        )
        mission_relevant = bool(gpt_result.get("mission_relevant", False))
        allowed_scene_types = {"kitchen", "market", "workshop", "clinic", "cleaning_room"}
        scene_type = str(gpt_result.get("scene_type", "")).strip().lower()
        if not mission_relevant or scene_type not in allowed_scene_types:
            scene_type = ""

        object_reports = gpt_result.get("object_reports")
        if isinstance(object_reports, list) and object_reports:
            scan_id = str(context.get("scan_id", self.scan_id or ""))
            for object_report in object_reports:
                if not isinstance(object_report, dict):
                    continue
                label = str(object_report.get("label", "")).strip().lower()
                if not label:
                    continue
                key = (scan_id, label)
                self.pending_object_reports[key] = {
                    "group_id": str(object_report.get("group_id", "group03_6769")),
                    "task_id": "M01",
                    "task_type": "object",
                    "scene_type": str(object_report.get("scene_type", scene_type)),
                    "label": label,
                    "pos": None,
                    "message": str(object_report.get("message", "")),
                }
                self.publish_pending_object_report(key)
        qr_report = gpt_result.get("qr_report")
        if isinstance(qr_report, dict):
            report = {
                "group_id": str(qr_report.get("group_id", self.get_parameter("report_group_id").value)),
                "task_id": str(qr_report.get("task_id", "M03")),
                "task_type": str(qr_report.get("task_type", "survivor")),
                "scene_type": str(qr_report.get("scene_type", "")),
                "label": str(qr_report.get("label", "")),
                "pos": qr_report.get("pos", [2.04, 3.06]),
                "message": str(qr_report.get("message", "")),
            }
        elif isinstance(object_reports, list) and object_reports:
            report = None
        else:
            report = {
                "group_id": str(self.get_parameter("report_group_id").value),
                "task_id": str(self.get_parameter("report_task_id").value),
                "task_type": str(self.get_parameter("report_task_type").value),
                "scene_type": scene_type,
                "label": str(gpt_result.get("label", "")) if mission_relevant else "",
                "pos": gpt_result.get("pos") if mission_relevant else None,
                "message": str(gpt_result.get("message", "")) if mission_relevant else "",
            }
        if report is not None:
            self.get_logger().info("Mission report: %s", json.dumps(report, ensure_ascii=False))
            self.publish_json(self.result_publisher, report)

        result = {
            **context,
            **gpt_result,
            "mission_relevant": mission_relevant,
            "qr_present": bool(gpt_result.get("qr_present", False))
            or bool(context["local_qr_detected"]),
            "hazards": hazards,
            "report": report,
        }
        if (result.get("qr_present") or hazards) and not self.approach_requested:
            self.approach_requested = True
            self.publish_json(self.locomotion_publisher, {
                "command": "approach_target",
                "scan_id": self.scan_id,
                "distance_meters": float(self.get_parameter("approach_distance_meters").value),
                "reason": {"qr": bool(result.get("qr_present")), "hazards": hazards},
            })

    def publish_hazard_view(
        self,
        jpeg: bytes,
        stamp: dict[str, int],
        scan_id: str,
        angle: float,
        publish_qr: bool,
    ) -> None:
        """Forward a scan view to hazard detection and optionally QR detection."""
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        try:
            message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning("Could not prepare hazard view: %s", error)
            return
        message.header.stamp.sec = int(stamp.get("sec", 0))
        message.header.stamp.nanosec = int(stamp.get("nanosec", 0))
        message.header.frame_id = f"{scan_id}|{angle}"
        self.hazard_view_publisher.publish(message)
        if publish_qr:
            self.qr_view_publisher.publish(message)

    @staticmethod
    def publish_json(publisher: Any, payload: dict[str, Any]) -> None:
        message = String()
        message.data = json.dumps(payload)
        publisher.publish(message)

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SceneUnderstandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
