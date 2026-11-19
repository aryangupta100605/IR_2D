"""Generate a synthetic hazard-sign dataset and train a five-class YOLO model.

The repository contains clean 800x800 ISO 7010 pictograms.  This script turns
those pictograms into object-detection examples by placing them on generated
industrial backgrounds and applying perspective, lighting, blur, noise, and
partial-occlusion effects.  The original signs are never modified.

Examples (run from the repository root)::

    python hazard_training/train_hazard_yolo.py prepare --samples-per-sign 250
    python hazard_training/train_hazard_yolo.py train --epochs 80
    python hazard_training/train_hazard_yolo.py all --samples-per-sign 250 --epochs 80

The five class IDs are deliberately fixed and must not be reordered without
also updating downstream ROS consumers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = [
    "high_voltage",
    "radioactive",
    "chemical",
    "biological",
    "flammable",
]

# ISO 7010 source sign -> requested output label.
SIGN_TO_CLASS = {
    "W003": "radioactive",  # Radioactive material or ionizing radiation
    "W009": "biological",  # Biological hazard
    "W012": "high_voltage",  # Electricity hazard
    "W016": "chemical",  # Toxic material
    "W021": "flammable",  # Flammable material
    "W023": "chemical",  # Corrosive substance
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SCRIPT_DIR = Path(__file__).resolve().parent


def find_source_signs(source_root: Path) -> dict[str, list[Path]]:
    """Find original images for the explicitly supported ISO sign IDs."""
    found: dict[str, list[Path]] = {sign: [] for sign in SIGN_TO_CLASS}
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem.upper()
        for sign in found:
            # Only use the clean originals.  Transformed files are generated
            # from the same source and would make the validation split leaky.
            if stem == f"ISO_7010_{sign}" and "original" in path.parent.name.lower():
                found[sign].append(path)
    missing = [sign for sign, paths in found.items() if not paths]
    if missing:
        raise FileNotFoundError(
            "Could not find original source pictograms for: " + ", ".join(missing)
        )
    return found


def random_background(rng: np.random.Generator, size: int) -> np.ndarray:
    """Create a textured, low-detail industrial background."""
    palette = np.array(
        [[105, 110, 112], [150, 145, 132], [74, 82, 88], [183, 174, 151]],
        dtype=np.float32,
    )
    base = palette[rng.integers(0, len(palette))]
    small = rng.normal(0, 18, (max(2, size // 16), max(2, size // 16), 1))
    texture = cv2.resize(small.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    background = np.clip(base + texture[..., None], 0, 255).astype(np.uint8)

    # Add seams, bolts, and stains so the model does not learn a single flat
    # background.  These are intentionally subtle and do not look like labels.
    for _ in range(int(rng.integers(3, 10))):
        color = tuple(int(x) for x in rng.integers(35, 110, 3))
        x1, y1 = rng.integers(0, size, 2)
        x2, y2 = rng.integers(0, size, 2)
        cv2.line(background, (int(x1), int(y1)), (int(x2), int(y2)), color, int(rng.integers(1, 5)))
    return background


def transform_sign(sign: np.ndarray, rng: np.random.Generator, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-transform a sign and return BGR pixels plus its mask."""
    sign = cv2.cvtColor(sign, cv2.COLOR_GRAY2BGR) if sign.ndim == 2 else sign
    h, w = sign.shape[:2]
    margin = 0.16
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    jitter = rng.uniform(-margin, margin, (4, 2)).astype(np.float32)
    destination = np.float32(
        [[w * (0.05 + jitter[0, 0]), h * (0.05 + jitter[0, 1])],
         [w * (0.95 + jitter[1, 0]), h * (0.05 + jitter[1, 1])],
         [w * (0.95 + jitter[2, 0]), h * (0.95 + jitter[2, 1])],
         [w * (0.05 + jitter[3, 0]), h * (0.95 + jitter[3, 1])]]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(sign, matrix, (w, h), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (w, h))
    scale = float(rng.uniform(0.22, 0.62) * output_size / max(h, w))
    new_size = max(24, int(w * scale)), max(24, int(h * scale))
    warped = cv2.resize(warped, new_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    return warped, mask


def make_example(sign: np.ndarray, rng: np.random.Generator, size: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Composite one sign into a scene and return a normalized YOLO box."""
    scene = random_background(rng, size)
    sign_image, sign_mask = transform_sign(sign, rng, size)
    sh, sw = sign_image.shape[:2]
    x = int(rng.integers(0, max(1, size - sw)))
    y = int(rng.integers(0, max(1, size - sh)))

    # Slightly translucent dirt/glare makes the sign less artificially clean.
    alpha = (sign_mask.astype(np.float32) / 255.0)[..., None]
    if rng.random() < 0.35:
        alpha *= float(rng.uniform(0.72, 0.96))
    roi = scene[y : y + sh, x : x + sw].astype(np.float32)
    scene[y : y + sh, x : x + sw] = np.clip(roi * (1 - alpha) + sign_image * alpha, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        scene = cv2.GaussianBlur(scene, (3, 3), float(rng.uniform(0.2, 1.1)))
    if rng.random() < 0.4:
        brightness = float(rng.uniform(-28, 24))
        scene = np.clip(scene.astype(np.float32) + brightness, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        noise = rng.normal(0, 5, scene.shape).astype(np.float32)
        scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # The mask's non-zero extent is the object box after perspective warp.
    ys, xs = np.where(sign_mask > 12)
    x1, x2 = x + int(xs.min()), x + int(xs.max()) + 1
    y1, y2 = y + int(ys.min()), y + int(ys.max()) + 1
    box = ((x1 + x2) / 2 / size, (y1 + y2) / 2 / size, (x2 - x1) / size, (y2 - y1) / size)
    return scene, box


def write_dataset(source_root: Path, output_root: Path, samples_per_sign: int, image_size: int, seed: int) -> None:
    """Generate a YOLO directory and data.yaml from the clean source signs."""
    sources = find_source_signs(source_root)
    rng = np.random.default_rng(seed)
    class_id = {name: index for index, name in enumerate(CLASS_NAMES)}

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sign, paths in sources.items():
        source_image = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        if source_image is None:
            raise RuntimeError(f"OpenCV could not read {paths[0]}")
        label = SIGN_TO_CLASS[sign]
        # Hold out a fixed subset for each source sign.  Since all variants of
        # a source share its appearance, this avoids a misleading random split.
        val_count = max(1, int(round(samples_per_sign * 0.2)))
        for index in range(samples_per_sign):
            split = "val" if index < val_count else "train"
            image, box = make_example(source_image, rng, image_size)
            filename = f"{sign.lower()}_{index:05d}.jpg"
            cv2.imwrite(str(output_root / "images" / split / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with (output_root / "labels" / split / f"{Path(filename).stem}.txt").open("w", encoding="utf-8") as file:
                file.write(f"{class_id[label]} {' '.join(f'{value:.6f}' for value in box)}\n")

    # Use an absolute path because Ultralytics may resolve a relative `path`
    # against the caller's working directory rather than data.yaml's folder.
    yaml = f"path: {output_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" + "\n".join(
        f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)
    ) + "\n"
    (output_root / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"Generated {samples_per_sign * len(sources)} images in {output_root}")


def train(dataset_root: Path, model: str, epochs: int, image_size: int, batch: int, device: str | None) -> None:
    """Train Ultralytics YOLO using the generated data.yaml."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt") from error
    yolo = YOLO(model)
    yolo.train(
        data=str(dataset_root / "data.yaml"),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        project=str(dataset_root / "runs"),
        name="hazard_yolo",
        pretrained=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "all"))
    # Defaults are anchored to this file, not the caller's current directory.
    # This allows the command to be run from either the repository root or
    # directly inside hazard_training/.
    parser.add_argument("--source", type=Path, default=SCRIPT_DIR / "images")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "dataset")
    parser.add_argument("--samples-per-sign", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics checkpoint or local .pt file")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, 0, cuda:0, etc.; default lets Ultralytics choose")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_sign < 2:
        raise SystemExit("--samples-per-sign must be at least 2")
    if args.command in {"prepare", "all"}:
        write_dataset(args.source, args.output, args.samples_per_sign, args.image_size, args.seed)
    if args.command in {"train", "all"}:
        train(args.output, args.model, args.epochs, args.image_size, args.batch, args.device)


if __name__ == "__main__":
    main()
