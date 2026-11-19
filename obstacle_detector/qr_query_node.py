#!/usr/bin/env python3
"""Decrypt QR payloads and send their meaning to the GPT query node."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import cv2
import httpx
import numpy as np
import rclpy
from cryptography.fernet import Fernet, InvalidToken
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


MISSION_CLASSIFICATION_PROMPT = """
Classify each mission below into exactly one mission ID. Return JSON only in the
form {"missions": [{"mission": 1, "task_id": "M01", "reason": "..."}]}.
Use these definitions exactly:
M01: find an object in a particular location.
M02: find a hazard in a particular area.
M03: find a survivor in a room; interpret the survivor location from the
     description if a location is provided.
M04: report the message from a survivor's QR code.
M05: find another survivor using a clue from the previous survivor and report
     the survivor's location.
Each mission must receive one of M01, M02, M03, M04, or M05. Return results
in the same order as the input and ignore numbering inside mission text.
Give a short reason.

Missions:
{missions}
""".strip()


class QrQueryNode(Node):
    """Scan camera frames and submit decrypted QR text for GPT interpretation."""

    GROUP_ID = "group03_6769"
    TASK_ID = "M05"
    TASK_TYPE = "survivor"

    def __init__(self) -> None:
        super().__init__("qr_query_node")
        self.declare_parameter("camera_topic", "/camerafeedback")
        self.declare_parameter("gpt_topic", "/gpt/query")
        self.declare_parameter("missions_file", "missions.txt")
        self.declare_parameter("mission_server", "http://192.168.10.141:8000")
        self.declare_parameter("mission_group", "group03_6769")
        self.declare_parameter("mission_timeout", 75.0)

        key = os.environ.get("QR_FERNET_KEY", "")
        if not key:
            raise RuntimeError("QR_FERNET_KEY is not set")
        try:
            self.fernet = Fernet(key.encode("ascii"))
        except (ValueError, TypeError, UnicodeEncodeError) as exc:
            raise RuntimeError("QR_FERNET_KEY is not a valid Fernet key") from exc

        camera_topic = str(self.get_parameter("camera_topic").value)
        gpt_topic = str(self.get_parameter("gpt_topic").value)
        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        self.last_token: str | None = None
        self.gpt_publisher = self.create_publisher(String, gpt_topic, 10)
        self.subscription = self.create_subscription(
            Image, camera_topic, self.image_callback, 10
        )
        self.get_logger().info(
            "Scanning %s for encrypted QR codes; sending QR meanings to GPT on %s",
            camera_topic,
            gpt_topic,
        )
        self.classify_startup_missions()

    def classify_startup_missions(self) -> None:
        """Read the initial mission set, classify it with GPT, and print it."""
        filename = Path(str(self.get_parameter("missions_file").value)).expanduser()
        try:
            missions = [
                line.strip()
                for line in filename.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        except OSError as exc:
            self.get_logger().warning("Could not read missions file %s: %s", filename, exc)
            return
        if not missions:
            self.get_logger().warning("Missions file is empty: %s", filename)
            return

        prompt = MISSION_CLASSIFICATION_PROMPT.format(
            missions="\n".join(
                f"{index}. {mission}" for index, mission in enumerate(missions, 1)
            )
        )
        try:
            encoded, buffer = cv2.imencode(".jpg", np.zeros((1, 1, 3), dtype=np.uint8))
            if not encoded:
                raise ValueError("could not create a GPT request image")
            image = buffer.tobytes()
            response = httpx.post(
                str(self.get_parameter("mission_server").value).rstrip("/") + "/analyze",
                data={
                    "group_id": str(self.get_parameter("mission_group").value),
                    "prompt": prompt,
                },
                files={"image": ("missions.jpg", image, "image/jpeg")},
                timeout=httpx.Timeout(float(self.get_parameter("mission_timeout").value), connect=5),
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("result"), str):
                payload = json.loads(payload["result"])
            classifications = payload.get("missions") if isinstance(payload, dict) else None
            if not isinstance(classifications, list):
                raise ValueError("GPT response does not contain a missions list")
        except (OSError, ValueError, TypeError, httpx.HTTPError, json.JSONDecodeError) as exc:
            self.get_logger().error("Mission classification failed: %s", exc)
            return

        print("Initial mission classification:")
        for index, mission in enumerate(missions, 1):
            item = classifications[index - 1] if index <= len(classifications) else {}
            if not isinstance(item, dict):
                item = {}
            task_id = str(item.get("task_id", "UNKNOWN")).upper()
            if task_id not in {"M01", "M02", "M03", "M04", "M05"}:
                task_id = "UNKNOWN"
            reason = str(item.get("reason", "")).strip()
            print(f"Mission {index} -> {task_id}: {reason or mission}")

    def image_callback(self, image_msg: Image) -> None:
        """Decode at most one QR code from each incoming camera frame."""
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error("Could not convert camera image: %s", exc)
            return

        token, _, _ = self.qr_detector.detectAndDecode(frame)
        if not token or token == self.last_token:
            return
        self.last_token = token
        try:
            decrypted = self.fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeEncodeError, UnicodeDecodeError) as exc:
            self.get_logger().warning("Could not decrypt QR payload: %s", exc)
            return

        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            self.get_logger().warning("Could not encode QR frame for GPT")
            return

        query = {
            "image_base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
            "context": {
                "qr_message": decrypted,
                "qr_token": token,
                "qr_task_id": self.TASK_ID,
                "qr_task_type": self.TASK_TYPE,
            },
        }
        message = String()
        message.data = json.dumps(query, ensure_ascii=False)
        self.gpt_publisher.publish(message)
        self.get_logger().info("Sent decrypted QR message to GPT for interpretation")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    try:
        node = QrQueryNode()
    except RuntimeError as exc:
        rclpy.logging.get_logger("qr_query_node").fatal(str(exc))
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
