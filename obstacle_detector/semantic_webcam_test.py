#!/usr/bin/env python3
"""Preview ADE20K YOLO detections and print significant objects."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


# ADE20K contains many structural/background classes that should not trigger
# the competition workflow when a room is otherwise empty.
WALL_FLOOR_CLASSES = {
    "wall", "floor", "ceiling", "sky", "building", "road", "path",
    "sidewalk", "earth", "mountain", "grass", "tree", "plant",
    "window", "door", "mirror", "light", "lamp",
}
CURTAIN_CLASSES = {"curtain", "blind"}
ADE_TO_MISSION_OBJECT = {
    "food": "food",
    "bottle": "water",
    "plate": "utensils",
    "pot": "utensils",
    "tray": "utensils",
    "person": "survivor",
}


def category_for(label: str) -> str:
    """Classify ADE20K labels into wall/floor, curtain, or important object."""
    if label in WALL_FLOOR_CLASSES:
        return "wall/floor"
    if label in CURTAIN_CLASSES:
        return "curtain"
    if label in ADE_TO_MISSION_OBJECT:
        return "important"
    return "ignore"


def infer_detections(
    model: object, frame: np.ndarray, confidence: float, imgsz: int
) -> list[tuple[str, str, float, tuple[int, int, int, int]]]:
    """Run one semantic inference and convert regions into display boxes."""
    results = model.predict(  # type: ignore[attr-defined]
        source=frame,
        conf=confidence,
        imgsz=imgsz,
        device="cpu",
        verbose=False,
    )
    detections: list[tuple[str, str, float, tuple[int, int, int, int]]] = []
    if not results:
        return detections
    result = results[0]
    if result.boxes is not None:
        for index in range(len(result.boxes)):
            class_id = int(result.boxes.cls[index].item())
            label = str(result.names[class_id]).strip().lower()
            confidence_value = float(result.boxes.conf[index].item())
            coordinates = tuple(int(value) for value in result.boxes.xyxy[index].tolist())
            category = category_for(label)
            if category != "ignore":
                detections.append((ADE_TO_MISSION_OBJECT.get(label, label), category, confidence_value, coordinates))
        return detections
    if result.semantic_mask is None:
        return detections

    mask = result.semantic_mask.data.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    scale_x = frame.shape[1] / mask.shape[1]
    scale_y = frame.shape[0] / mask.shape[0]
    for class_id in np.unique(mask):
        class_id = int(class_id)
        label = str(result.names.get(class_id, class_id)).strip().lower()
        class_mask = (mask == class_id).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(class_mask)
        for component in range(1, count):
            left, top, width, height, area = stats[component]
            if area < 100:
                continue
            category = category_for(label)
            if category != "ignore":
                detections.append(
                    (
                        ADE_TO_MISSION_OBJECT.get(label, label),
                        category,
                        1.0,
                        (
                            int(left * scale_x),
                            int(top * scale_y),
                            int((left + width) * scale_x),
                            int((top + height) * scale_y),
                        ),
                    )
                )
    return detections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run yolo26n-sem-ade20k on a webcam and print significant objects."
    )
    parser.add_argument("--source", default="0", help="Webcam index or video path.")
    parser.add_argument("--model", default="yolo26n-sem-ade20k.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument(
        "--inference-every",
        type=int,
        default=5,
        help="Run YOLO once every N camera frames (default: 5).",
    )
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args(argv)

    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    from ultralytics import YOLO

    model_path = Path(args.model).expanduser()
    print(f"Loading semantic model: {model_path}")
    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open webcam source: {args.source}")

    previous_objects: tuple[str, ...] = ()
    detections: list[tuple[str, str, float, tuple[int, int, int, int]]] = []
    frame_number = 0
    inference: Future[list[tuple[str, str, float, tuple[int, int, int, int]]]] | None = None
    executor = ThreadPoolExecutor(max_workers=1)
    print("Camera started. Press q or Esc to quit.")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Could not read a webcam frame.")
                break

            frame_number += 1
            if inference is not None and inference.done():
                detections = inference.result()
                inference = None
            if (
                inference is None
                and (frame_number == 1 or frame_number % max(args.inference_every, 1) == 0)
            ):
                inference = executor.submit(
                    infer_detections, model, frame.copy(), args.confidence, args.imgsz
                )

            for label, category, confidence, (left, top, right, bottom) in detections:
                    color = {
                        "wall/floor": (255, 0, 0),
                        "curtain": (0, 255, 255),
                        "important": (0, 255, 0),
                    }[category]
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {confidence:.2f}",
                        (left, max(top - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            wall_floor = tuple(sorted({label for label, category, _, _ in detections if category == "wall/floor"}))
            curtains = tuple(sorted({label for label, category, _, _ in detections if category == "curtain"}))
            important = tuple(sorted({label for label, category, _, _ in detections if category == "important"}))
            current_objects = wall_floor + ("|",) + curtains + ("|",) + important
            if current_objects != previous_objects:
                print("Wall/floor: " + (", ".join(wall_floor) or "none"))
                print("Curtain: " + (", ".join(curtains) or "none"))
                print("Important: " + (", ".join(important) or "none"))
                previous_objects = current_objects

            cv2.imshow("ADE20K semantic object test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
