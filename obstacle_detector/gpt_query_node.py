#!/usr/bin/env python3
"""Query the LAN GPT perception server when the overseer requests it."""

from __future__ import annotations

import base64
import argparse
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

import cv2
import httpx

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # Allow the webcam test mode to run without ROS 2.
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    String = Any  # type: ignore[misc,assignment]


DEFAULT_PROMPT = (
    "Analyze this camera image for a simulated post-disaster robotics competition. "
    "Return JSON only with exactly these keys: mission_relevant (boolean), "
    "scene_type (if mission_relevant, exactly one of kitchen, market, workshop, "
    "clinic, cleaning_room; otherwise an empty string), label (the most relevant "
    "object or supply label, or an empty string), objects (array containing zero "
    "or more of medical_kit, hand_tools, food, water, utensils), pos (a two-number "
    "position only if clearly provided by the image context, otherwise null), "
    "message (normally "
    "an empty string), humans (integer count of clearly visible people), "
    "description (a concise description of the scene), qr_present (boolean), "
    "hazard_present (boolean), and hazards (array of visible safety-sign labels). "
    "hazards may contain only high_voltage, radioactive, chemical, biological, "
    "flammable. Set hazard_present false and hazards to [] when no hazard is "
    "visibly available. Do not infer a "
    "hazard from ordinary objects. The environment may use miniature, toy, "
    "cardboard, or printed props to represent real objects, rooms, equipment, "
    "supplies, and other real-world items. Interpret these props according to the "
    "real-world objects or locations they clearly represent. Do not mark a scene "
    "irrelevant merely because the objects are toys, miniature, or nonfunctional "
    "props. If the scene is not relevant to the post-disaster mission, set "
    "mission_relevant to false and leave scene_type, label, and message empty."
)
DEFAULT_SERVER = "http://192.168.10.141:8000"
DEFAULT_QR_POSITION = [2.04, 3.06]
DEFAULT_QR_GROUP = "group03_6769"
ALLOWED_SCENE_TYPES = {"kitchen", "market", "workshop", "clinic", "cleaning_room"}
ALLOWED_OBJECT_LABELS = {"medical_kit", "hand_tools", "food", "water", "utensils"}
ALLOWED_HAZARD_LABELS = {
    "high_voltage",
    "radioactive",
    "chemical",
    "biological",
    "flammable",
}


def query_server_request(
    image: bytes, server: str, group: str, prompt: str, timeout: float
) -> dict[str, Any]:
    """Send one JPEG to the LAN perception server and return its JSON result."""

    upload = ("room_view.jpg", image, mimetypes.types_map[".jpg"])
    response = httpx.post(
        server.rstrip("/") + "/analyze",
        data={"group_id": group, "prompt": prompt},
        files={"image": upload},
        timeout=httpx.Timeout(timeout, connect=5),
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = response.text.strip()
        raise RuntimeError(f"{error}; server response: {detail}") from error
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("server response must be a JSON object")
    return payload


class GPTQueryNode(Node):
    """Perform one LAN perception request for each overseer trigger."""

    def __init__(self) -> None:
        super().__init__("gpt_query_node")

        self.declare_parameter("notification_topic", "/gpt/query")
        self.declare_parameter("result_topic", "/gpt/results")
        self.declare_parameter("qr_notification_topic", "/qr/notifications")
        self.declare_parameter("object_notification_topic", "/object/notifications")
        self.declare_parameter("qr_position", "2.04,3.06")
        self.declare_parameter("qr_group", DEFAULT_QR_GROUP)
        self.declare_parameter("server", DEFAULT_SERVER)
        self.declare_parameter("group", "group03_6769")
        self.declare_parameter("prompt", DEFAULT_PROMPT)
        self.declare_parameter("prompt_file", "")
        self.declare_parameter("read_timeout", 75.0)
        notification_topic = str(self.get_parameter("notification_topic").value)
        result_topic = str(self.get_parameter("result_topic").value)
        qr_notification_topic = str(
            self.get_parameter("qr_notification_topic").value
        )
        object_notification_topic = str(
            self.get_parameter("object_notification_topic").value
        )

        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.qr_notification_publisher = self.create_publisher(
            String, qr_notification_topic, 10
        )
        self.object_notification_publisher = self.create_publisher(
            String, object_notification_topic, 10
        )
        self.subscription = self.create_subscription(
            String,
            notification_topic,
            self.notification_callback,
            10,
        )
        self.get_logger().info(
            "Listening for GPT query triggers on %s; results on %s",
            notification_topic,
            result_topic,
        )

    def notification_callback(self, message: String) -> None:
        """Run one query after receiving a JSON trigger from the overseer."""
        try:
            request = json.loads(message.data)
            if not isinstance(request, dict):
                raise ValueError("query trigger must be a JSON object")
            image = base64.b64decode(request["image_base64"], validate=True)
            context = request.get("context", {})
            if not isinstance(context, dict):
                raise ValueError("query context must be a JSON object")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning("Ignoring invalid GPT query trigger: %s", error)
            return

        result = self.query_server(
            image=image,
            group=str(request.get("group", self.get_parameter("group").value)),
            prompt=self.load_prompt(request.get("prompt")),
        )
        result["hazards"] = self.visible_hazards(result)
        result["hazard_present"] = bool(result["hazards"])
        qr_found = bool(result.get("qr_present")) or bool(
            context.get("local_qr_detected")
        )
        if qr_found:
            position = self.qr_position()
            self.publish_qr_notification(context, position)
            scene_type = str(result.get("scene_type", "")).strip().lower()
            if scene_type not in ALLOWED_SCENE_TYPES:
                scene_type = ""
            result["qr_report"] = {
                "group_id": str(self.get_parameter("qr_group").value),
                "task_id": "M03",
                "task_type": "survivor",
                "scene_type": scene_type,
                "label": "",
                "pos": position,
                "message": "",
            }
        object_labels = self.object_labels(result.get("objects"))
        if not object_labels:
            fallback_label = str(result.get("label", "")).strip().lower()
            if fallback_label in ALLOWED_OBJECT_LABELS:
                object_labels = [fallback_label]
        if object_labels:
            scene_type = str(result.get("scene_type", "")).strip().lower()
            if scene_type not in ALLOWED_SCENE_TYPES:
                scene_type = ""
            self.publish_object_notification(context, object_labels)
            result["object_reports"] = [
                {
                    "group_id": DEFAULT_QR_GROUP,
                    "task_id": "M01",
                    "task_type": "object",
                    "scene_type": scene_type,
                    "label": label,
                    "pos": None,
                    "message": "",
                }
                for label in object_labels
            ]
        output = String()
        output.data = json.dumps({"context": context, "result": result})
        self.result_publisher.publish(output)

    @staticmethod
    def object_labels(value: Any) -> list[str]:
        """Keep only unique object labels from GPT's classified object list."""
        if not isinstance(value, list):
            return []
        labels: list[str] = []
        for item in value:
            label = str(item).strip().lower()
            if label in ALLOWED_OBJECT_LABELS and label not in labels:
                labels.append(label)
        return labels

    @staticmethod
    def visible_hazards(result: dict[str, Any]) -> list[str]:
        """Keep only explicitly visible, allowed hazard labels."""
        raw_hazards = result.get("hazards", [])
        if not bool(result.get("hazard_present", bool(raw_hazards))):
            return []
        if not isinstance(raw_hazards, list):
            return []
        return sorted(
            {
                str(label).strip().lower()
                for label in raw_hazards
                if str(label).strip().lower() in ALLOWED_HAZARD_LABELS
            }
        )

    def publish_object_notification(
        self, context: dict[str, Any], labels: list[str]
    ) -> None:
        """Ask the overseer/navigation stack for coordinates for each object."""
        message = String()
        message.data = json.dumps(
            {
                "event": "objects_detected",
                "detected": True,
                "scan_id": context.get("scan_id", ""),
                "labels": labels,
                "request_coordinates": True,
            }
        )
        self.object_notification_publisher.publish(message)

    def qr_position(self) -> list[float]:
        """Return the provisional navigation position for a scanned QR code."""
        raw_position = str(self.get_parameter("qr_position").value)
        try:
            position = [float(value.strip()) for value in raw_position.split(",")]
        except ValueError:
            position = []
        if len(position) != 2:
            self.get_logger().warning(
                "Invalid qr_position %r; using default %s",
                raw_position,
                DEFAULT_QR_POSITION,
            )
            return list(DEFAULT_QR_POSITION)
        return position

    def publish_qr_notification(
        self, context: dict[str, Any], position: list[float]
    ) -> None:
        """Tell the overseer that a QR was scanned and coordinates are needed."""
        message = String()
        message.data = json.dumps(
            {
                "event": "qr_scanned",
                "detected": True,
                "scan_id": context.get("scan_id", ""),
                "request_coordinates": True,
                "pos": position,
            }
        )
        self.qr_notification_publisher.publish(message)

    def load_prompt(self, requested_prompt: Any) -> str:
        """Use a trigger prompt, an optional file, or the built-in prompt."""
        if isinstance(requested_prompt, str) and requested_prompt.strip():
            return requested_prompt
        prompt_file = str(self.get_parameter("prompt_file").value).strip()
        if prompt_file:
            try:
                prompt = Path(prompt_file).read_text(encoding="utf-8-sig")
                if prompt.strip():
                    return prompt
                self.get_logger().warning("Prompt file is blank: %s", prompt_file)
            except OSError as error:
                self.get_logger().warning("Could not read prompt file %s: %s", prompt_file, error)
        return str(self.get_parameter("prompt").value) or DEFAULT_PROMPT

    def query_server(self, image: bytes, group: str, prompt: str) -> dict[str, Any]:
        """Copy the test client's multipart LAN request behavior."""
        server = str(self.get_parameter("server").value).rstrip("/")
        timeout = float(self.get_parameter("read_timeout").value)
        try:
            return query_server_request(image, server, group, prompt, timeout)
        except (OSError, RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error("GPT query failed: %s", error)
            return {"qr_present": False, "hazards": [], "notes": "gpt_error"}


def webcam_test_main(argv: list[str]) -> int:
    """Preview a webcam and query the current frame when Space is pressed."""

    parser = argparse.ArgumentParser(description="Query the LAN GPT server from a webcam.")
    parser.add_argument("--webcam-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", default="0", help="Webcam index (default: 0).")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--group", default="group03_6769")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument(
        "--human-model",
        default=str(Path(__file__).resolve().parents[1] / "yolo11n.pt"),
        help="YOLO weights used for local person detection.",
    )
    parser.add_argument("--human-confidence", type=float, default=0.35)
    parser.add_argument("--human-iou", type=float, default=0.45)
    args = parser.parse_args(argv)

    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    try:
        from ultralytics import YOLO
        from obstacle_detector.windows_detector_test import detect_humans, detect_qr
    except ImportError as error:
        raise SystemExit(
            "Install dependencies first: python -m pip install -r requirements.txt"
        ) from error

    human_model_path = Path(args.human_model).expanduser()
    if not human_model_path.exists():
        raise SystemExit(f"Human model not found: {human_model_path}")
    print(f"Loading human model: {human_model_path}")
    human_model = YOLO(str(human_model_path))
    qr_detector = cv2.QRCodeDetector()
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open webcam source: {args.source}")

    print(f"Webcam test connected. Server: {args.server.rstrip('/')}/analyze")
    print("Press Space to query a frame; press q or Esc to quit.")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Could not read a webcam frame.")
                break
            cv2.imshow("GPT webcam test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    print("Could not encode the current frame.")
                    continue
                print("Querying GPT...")
                try:
                    result = query_server_request(
                        buffer.tobytes(),
                        args.server,
                        args.group,
                        args.prompt,
                        args.timeout,
                    )
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                except (OSError, RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
                    print(f"GPT query failed: {error}")

                print("Running local person detector...")
                humans = detect_humans(
                    human_model, frame, args.human_confidence, args.human_iou, ""
                )
                print(
                    f"Detected {len(humans)} person(s). "
                    "Person bounding boxes are shown in the preview window."
                )

                print("Running local QR detector...")
                qrs = detect_qr(frame, qr_detector)
                qr_payloads = [d["payload"] for d in qrs if d.get("payload")]
                print(
                    f"Detected {len(qrs)} QR code(s). "
                    "QR bounding boxes are shown in the preview window."
                )
                if qr_payloads:
                    print(json.dumps({"qr_payloads": qr_payloads}, ensure_ascii=False))
                cv2.imshow("GPT webcam test", frame)
                cv2.waitKey(1)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main(args: list[str] | None = None) -> None:
    command_args = sys.argv[1:] if args is None else args
    if "--webcam-test" in command_args:
        raise SystemExit(webcam_test_main(command_args))
    if rclpy is None:
        raise SystemExit("ROS 2 is not installed. Use --webcam-test for webcam-only testing.")
    rclpy.init(args=args)
    node = GPTQueryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
