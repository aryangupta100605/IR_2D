#!/usr/bin/env python3
"""Forward JSON messages from robot subsystems to the task-report server.

This node deliberately does not define the JSON schema for task messages.
The QR and GPT nodes are responsible for producing the JSON string; this node
validates that it is JSON and forwards the original bytes unchanged.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskReportNode(Node):
    """Listen for QR/GPT reports and send them to the reporting server."""

    def __init__(self) -> None:
        super().__init__("task_report_node")

        self.declare_parameter("task_report_url", "http://192.168.10.141:8001/task_report")
        self.declare_parameter("qr_topic", "/qr/decrypted")
        self.declare_parameter("gpt_topic", "/scene/understanding")
        self.declare_parameter("request_timeout_seconds", 5.0)
        self.declare_parameter("report_startup", True)
        self.declare_parameter("send_reports", True)

        self.report_url = str(self.get_parameter("task_report_url").value)
        self.request_timeout = float(
            self.get_parameter("request_timeout_seconds").value
        )
        self.send_reports = bool(self.get_parameter("send_reports").value)
        self.executor = ThreadPoolExecutor(max_workers=2)

        qr_topic = str(self.get_parameter("qr_topic").value)
        gpt_topic = str(self.get_parameter("gpt_topic").value)
        self.qr_subscription = self.create_subscription(
            String,
            qr_topic,
            lambda message: self.message_callback(message, "qr"),
            10,
        )
        self.gpt_subscription = self.create_subscription(
            String,
            gpt_topic,
            lambda message: self.message_callback(message, "gpt"),
            10,
        )

        self.get_logger().info(
            "%s JSON from %s and %s to %s",
            "Sending" if self.send_reports else "Print-only",
            qr_topic,
            gpt_topic,
            self.report_url,
        )

        if bool(self.get_parameter("report_startup").value):
            startup_message = json.dumps({"event": "robot_initialized"})
            self.executor.submit(self.send_json, startup_message, "startup")

    def message_callback(self, message: String, source: str) -> None:
        """Queue a raw JSON message without changing its schema."""
        try:
            json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(
                "Ignoring invalid JSON received from %s: %s", source, exc
            )
            return

        self.executor.submit(self.send_json, message.data, source)

    def send_json(self, body: str, source: str) -> None:
        """POST the original JSON body to the configured reporting server."""
        self.get_logger().info(
            "Report body (%s): %s", source, body
        )
        if not self.send_reports:
            self.get_logger().info(
                "Print-only mode: skipped HTTP request for %s report", source
            )
            return
        request = Request(
            self.report_url,
            data=body.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Report-Source": source,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                self.get_logger().info(
                    "Task server response (%s HTTP %s): %s",
                    source,
                    response.status,
                    response_body or "<empty>",
                )
            self.get_logger().info("Sent %s report to task server", source)
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            self.get_logger().error(
                "Task server rejected %s report with HTTP %s: %s",
                source,
                exc.code,
                response_body or "<empty>",
            )
        except URLError as exc:
            self.get_logger().error("Could not reach task server for %s report: %s", source, exc.reason)
        except TimeoutError:
            self.get_logger().error("Task server timed out for %s report", source)

    def close(self) -> None:
        """Stop outstanding HTTP worker threads during node shutdown."""
        self.executor.shutdown(wait=True)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TaskReportNode()
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
