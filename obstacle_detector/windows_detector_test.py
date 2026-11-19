#!/usr/bin/env python3
"""Run human, hazard, and QR detectors locally without ROS 2.

Examples (PowerShell)::

    python -m obstacle_detector.windows_detector_test --source 0
    python -m obstacle_detector.windows_detector_test --source .\test.jpg
    python -m obstacle_detector.windows_detector_test --source 0 --no-display --frames 1

Press ``q`` or ``Esc`` in the preview window to stop. QR detection is only
run for frames where the human model detects a person.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


HAZARD_CLASSES = {
    "high_voltage",
    "radioactive",
    "chemical",
    "biological",
    "flammable",
}
DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1] / "hazard_detection" / "weights" / "best.pt"
)
DEFAULT_HUMAN_MODEL = Path(__file__).resolve().parents[1] / "yolo11n.pt"


def parse_source(value: str) -> int | str:
    """Treat a numeric source as a webcam index, otherwise as a file path."""

    try:
        return int(value)
    except ValueError:
        return value


def draw_label(
    frame: Any, text: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    cv2.putText(
        frame,
        text,
        (max(x, 0), max(y, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def detect_hazards(
    model: Any, frame: Any, confidence: float, iou: float, device: str
) -> list[dict[str, Any]]:
    """Return hazard detections and draw their red boxes on ``frame``."""

    kwargs: dict[str, Any] = {
        "source": frame,
        "conf": confidence,
        "iou": iou,
        "max_det": 20,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device

    result_list = model.predict(**kwargs)
    if not result_list:
        return []

    result = result_list[0]
    detections: list[dict[str, Any]] = []
    for index in range(len(result.boxes)):
        class_id = int(result.boxes.cls[index].item())
        label = str(result.names[class_id]).strip().lower()
        if label not in HAZARD_CLASSES:
            continue

        confidence_value = float(result.boxes.conf[index].item())
        left, top, right, bottom = (
            int(value) for value in result.boxes.xyxy[index].tolist()
        )
        draw_label(
            frame,
            f"HAZARD: {label} {confidence_value:.2f}",
            left,
            top - 8,
            (0, 0, 255),
        )
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        detections.append(
            {
                "label": label,
                "confidence": confidence_value,
                "bbox": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
            }
        )
    return detections


def detect_humans(
    model: Any, frame: Any, confidence: float, iou: float, device: str
) -> list[dict[str, Any]]:
    """Return person detections and draw their blue boxes on ``frame``."""

    kwargs: dict[str, Any] = {
        "source": frame,
        "conf": confidence,
        "iou": iou,
        "max_det": 20,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device

    result_list = model.predict(**kwargs)
    if not result_list:
        return []

    result = result_list[0]
    detections: list[dict[str, Any]] = []
    for index in range(len(result.boxes)):
        class_id = int(result.boxes.cls[index].item())
        label = str(result.names[class_id]).strip().lower()
        if label != "person":
            continue

        confidence_value = float(result.boxes.conf[index].item())
        left, top, right, bottom = (
            int(value) for value in result.boxes.xyxy[index].tolist()
        )
        draw_label(
            frame,
            f"HUMAN: {confidence_value:.2f}",
            left,
            top - 8,
            (255, 0, 0),
        )
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        detections.append(
            {
                "label": label,
                "confidence": confidence_value,
                "bbox": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
            }
        )
    return detections


def detect_qr(frame: Any, detector: Any) -> list[dict[str, Any]]:
    """Return QR detections and draw their green boxes on ``frame``."""

    detections: list[dict[str, Any]] = []
    multi_result = detector.detectAndDecodeMulti(frame)
    if len(multi_result) == 4:
        found, decoded_info, points, _ = multi_result
        if found and points is not None:
            for payload, corners in zip(decoded_info, points):
                detections.append(_draw_qr(frame, payload, corners))
            return detections

    payload, points, _ = detector.detectAndDecode(frame)
    if points is not None:
        detections.append(_draw_qr(frame, payload, points[0]))
    return detections


def _draw_qr(frame: Any, payload: str, corners: Any) -> dict[str, Any]:
    corners = corners.astype(int)
    for index in range(4):
        cv2.line(
            frame,
            tuple(corners[index]),
            tuple(corners[(index + 1) % 4]),
            (0, 255, 0),
            2,
        )

    left = int(corners[:, 0].min())
    top = int(corners[:, 1].min())
    label = "QR detected" if not payload else f"QR: {payload[:32]}"
    draw_label(frame, label, left, top - 8, (0, 255, 0))
    return {
        "payload": payload,
        "bbox": {
            "left": left,
            "top": top,
            "right": int(corners[:, 0].max()),
            "bottom": int(corners[:, 1].max()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test human-gated hazard and QR detection from a webcam, image, or video without ROS."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (default: 0), or an image/video path.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Hazard YOLO weights (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--human-model",
        default=str(DEFAULT_HUMAN_MODEL),
        help=f"Human YOLO weights (default: {DEFAULT_HUMAN_MODEL}).",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--human-confidence", type=float, default=0.35)
    parser.add_argument("--human-iou", type=float, default=0.45)
    parser.add_argument("--device", default="", help="Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--window", default="QR + hazard detector test")
    parser.add_argument("--save", type=Path, help="Save the annotated output image/video.")
    parser.add_argument("--json", action="store_true", help="Print detections as JSON.")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV window.")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 means until q/Esc or the source ends.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise SystemExit(f"Hazard model not found: {model_path}")
    human_model_path = Path(args.human_model).expanduser()
    if not human_model_path.exists():
        raise SystemExit(f"Human model not found: {human_model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt") from error

    print(f"Loading human model: {human_model_path}")
    human_model = YOLO(str(human_model_path))
    print(f"Loading hazard model: {model_path}")
    model = YOLO(str(model_path))
    qr_detector = cv2.QRCodeDetector()
    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    is_image = isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
    }
    writer: Any = None
    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            humans = detect_humans(
                human_model,
                frame,
                args.human_confidence,
                args.human_iou,
                args.device,
            )
            hazards = detect_hazards(model, frame, args.confidence, args.iou, args.device)
            qrs = detect_qr(frame, qr_detector) if humans else []
            frame_count += 1
            if args.json:
                print(
                    json.dumps(
                        {
                            "frame": frame_count,
                            "humans": humans,
                            "hazards": hazards,
                            "qr_triggered": bool(humans),
                            "qr": qrs,
                        }
                    )
                )

            if args.save:
                if writer is None:
                    height, width = frame.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
                    suffix = args.save.suffix.lower()
                    if is_image or suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        cv2.imwrite(str(args.save), frame)
                    else:
                        writer = cv2.VideoWriter(
                            str(args.save),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height),
                        )
                if writer is not None:
                    writer.write(frame)

            if not args.no_display:
                cv2.imshow(args.window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.frames > 0 and frame_count >= args.frames:
                break
            if is_image:
                if args.no_display:
                    break
                cv2.waitKey(0)
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
