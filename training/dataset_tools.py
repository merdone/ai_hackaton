from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetRoot:
    root: Path
    images_dir: Path
    labels_dir: Path
    classes_path: Path
    dataset_yaml_path: Path | None
    notes_path: Path | None


@dataclass(frozen=True)
class DatasetSummary:
    class_names: list[str]
    num_images: int
    num_labels: int
    missing_labels_count: int
    extra_labels_count: int
    empty_labels_count: int
    malformed_count: int
    out_of_bounds_count: int
    invalid_class_count: int
    zero_area_count: int
    corrupt_images_count: int
    image_width_range: tuple[int, int] | None
    image_height_range: tuple[int, int] | None
    unique_sizes_count: int
    boxes_total: int
    boxes_per_image_mean: float
    boxes_per_image_median: float
    max_boxes_in_one_image: int
    sample_missing_labels: list[str]
    sample_extra_labels: list[str]
    sample_empty_labels: list[str]
    sample_malformed: list[tuple[str, int, str]]
    sample_out_of_bounds: list[tuple[str, int, list[float]]]
    sample_invalid_class: list[tuple[str, int, int]]
    sample_zero_area: list[tuple[str, int, list[float]]]
    sample_corrupt_images: list[tuple[str, str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "class_names": self.class_names,
            "num_images": self.num_images,
            "num_labels": self.num_labels,
            "missing_labels_count": self.missing_labels_count,
            "extra_labels_count": self.extra_labels_count,
            "empty_labels_count": self.empty_labels_count,
            "malformed_count": self.malformed_count,
            "out_of_bounds_count": self.out_of_bounds_count,
            "invalid_class_count": self.invalid_class_count,
            "zero_area_count": self.zero_area_count,
            "corrupt_images_count": self.corrupt_images_count,
            "image_width_range": list(self.image_width_range) if self.image_width_range else None,
            "image_height_range": list(self.image_height_range) if self.image_height_range else None,
            "unique_sizes_count": self.unique_sizes_count,
            "boxes_total": self.boxes_total,
            "boxes_per_image_mean": self.boxes_per_image_mean,
            "boxes_per_image_median": self.boxes_per_image_median,
            "max_boxes_in_one_image": self.max_boxes_in_one_image,
            "sample_missing_labels": self.sample_missing_labels,
            "sample_extra_labels": self.sample_extra_labels,
            "sample_empty_labels": self.sample_empty_labels,
            "sample_malformed": self.sample_malformed,
            "sample_out_of_bounds": self.sample_out_of_bounds,
            "sample_invalid_class": self.sample_invalid_class,
            "sample_zero_area": self.sample_zero_area,
            "sample_corrupt_images": self.sample_corrupt_images,
        }


def resolve_dataset_root(source_root: Path) -> DatasetRoot:
    root = source_root.expanduser().resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    classes_path = root / "classes.txt"
    dataset_yaml_path = root / "dataset.yaml"
    notes_path = root / "notes.json"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")
    if not classes_path.is_file():
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")

    return DatasetRoot(
        root=root,
        images_dir=images_dir,
        labels_dir=labels_dir,
        classes_path=classes_path,
        dataset_yaml_path=dataset_yaml_path if dataset_yaml_path.is_file() else None,
        notes_path=notes_path if notes_path.is_file() else None,
    )


def resolve_split_dirs(dataset_root: Path, split_name: str) -> tuple[Path, Path]:
    split_images = dataset_root / split_name / "images"
    split_labels = dataset_root / split_name / "labels"
    if split_images.is_dir() and split_labels.is_dir():
        return split_images, split_labels

    return dataset_root / "images", dataset_root / "labels"


def load_class_names(dataset: DatasetRoot) -> list[str]:
    return [
        line.strip()
        for line in dataset.classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iter_image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image_unicode(path: Path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    if encoded.size == 0:
        return None

    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def read_label_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []

    raw_text = label_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def validate_dataset(dataset: DatasetRoot) -> DatasetSummary:
    class_names = load_class_names(dataset)
    images = iter_image_paths(dataset.images_dir)
    labels = sorted(dataset.labels_dir.glob("*.txt"))
    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels}
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)

    empty_labels: list[str] = []
    malformed: list[tuple[str, int, str]] = []
    out_of_bounds: list[tuple[str, int, list[float]]] = []
    invalid_class: list[tuple[str, int, int]] = []
    zero_area: list[tuple[str, int, list[float]]] = []
    corrupt_images: list[tuple[str, str]] = []
    box_counts: list[int] = []
    image_sizes: list[tuple[int, int]] = []

    for image_path in images:
        image = read_image_unicode(image_path)
        if image is None:
            corrupt_images.append((image_path.name, "imdecode returned None"))
            continue

        height, width = image.shape[:2]
        image_sizes.append((width, height))

    for label_path in labels:
        lines = read_label_lines(label_path)
        if not lines:
            empty_labels.append(label_path.name)
            box_counts.append(0)
            continue

        box_counts.append(len(lines))
        for line_index, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                malformed.append((label_path.name, line_index, line))
                continue

            try:
                class_id = int(float(parts[0]))
                center_x, center_y, width, height = map(float, parts[1:])
            except ValueError:
                malformed.append((label_path.name, line_index, line))
                continue

            if class_id < 0 or class_id >= len(class_names):
                invalid_class.append((label_path.name, line_index, class_id))

            values = [center_x, center_y, width, height]
            if any(value < 0 or value > 1 for value in values):
                out_of_bounds.append((label_path.name, line_index, values))
            if width <= 0 or height <= 0:
                zero_area.append((label_path.name, line_index, values))

    width_range = None
    height_range = None
    if image_sizes:
        widths = [width for width, _ in image_sizes]
        heights = [height for _, height in image_sizes]
        width_range = (min(widths), max(widths))
        height_range = (min(heights), max(heights))

    boxes_total = sum(box_counts)
    boxes_per_image_mean = round(boxes_total / len(box_counts), 3) if box_counts else 0.0
    sorted_box_counts = sorted(box_counts)
    if not sorted_box_counts:
        boxes_per_image_median = 0.0
    else:
        middle = len(sorted_box_counts) // 2
        if len(sorted_box_counts) % 2 == 1:
            boxes_per_image_median = float(sorted_box_counts[middle])
        else:
            boxes_per_image_median = (
                sorted_box_counts[middle - 1] + sorted_box_counts[middle]
            ) / 2.0

    return DatasetSummary(
        class_names=class_names,
        num_images=len(images),
        num_labels=len(labels),
        missing_labels_count=len(missing_labels),
        extra_labels_count=len(extra_labels),
        empty_labels_count=len(empty_labels),
        malformed_count=len(malformed),
        out_of_bounds_count=len(out_of_bounds),
        invalid_class_count=len(invalid_class),
        zero_area_count=len(zero_area),
        corrupt_images_count=len(corrupt_images),
        image_width_range=width_range,
        image_height_range=height_range,
        unique_sizes_count=len(set(image_sizes)),
        boxes_total=boxes_total,
        boxes_per_image_mean=boxes_per_image_mean,
        boxes_per_image_median=boxes_per_image_median,
        max_boxes_in_one_image=max(box_counts) if box_counts else 0,
        sample_missing_labels=missing_labels[:10],
        sample_extra_labels=extra_labels[:10],
        sample_empty_labels=empty_labels[:10],
        sample_malformed=malformed[:5],
        sample_out_of_bounds=out_of_bounds[:5],
        sample_invalid_class=invalid_class[:5],
        sample_zero_area=zero_area[:5],
        sample_corrupt_images=corrupt_images[:5],
    )


def _draw_ground_truth(image, label_path: Path):
    lines = read_label_lines(label_path)
    if not lines:
        return image

    height, width = image.shape[:2]
    for line in lines:
        _, center_x, center_y, box_width, box_height = line.split()
        center_x = float(center_x) * width
        center_y = float(center_y) * height
        box_width = float(box_width) * width
        box_height = float(box_height) * height
        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return image


def create_contact_sheet(
    dataset: DatasetRoot,
    output_path: Path,
    sample_size: int = 12,
    seed: int = 7,
) -> Path:
    image_paths = iter_image_paths(dataset.images_dir)
    if not image_paths:
        raise RuntimeError("Dataset has no images.")

    randomizer = random.Random(seed)
    sample = image_paths if len(image_paths) <= sample_size else randomizer.sample(image_paths, sample_size)
    tiles = []
    for image_path in sample:
        image = read_image_unicode(image_path)
        if image is None:
            continue
        rendered = _draw_ground_truth(image, dataset.labels_dir / f"{image_path.stem}.txt")
        thumbnail = cv2.resize(rendered, (480, 270))
        cv2.putText(
            thumbnail,
            image_path.name,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        tiles.append(thumbnail)

    if not tiles:
        raise RuntimeError("Could not render any dataset samples.")

    while len(tiles) % 3 != 0:
        tiles.append(np.zeros_like(tiles[0]))

    rows = [
        cv2.hconcat(tiles[index:index + 3])
        for index in range(0, len(tiles), 3)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.vconcat(rows))
    return output_path


def split_dataset(
    dataset: DatasetRoot,
    output_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output folder already exists and is not empty: {output_root}. "
            "Pass --overwrite to recreate it."
        )

    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)

    positive_images: list[Path] = []
    negative_images: list[Path] = []
    for image_path in iter_image_paths(dataset.images_dir):
        label_lines = read_label_lines(dataset.labels_dir / f"{image_path.stem}.txt")
        if label_lines:
            positive_images.append(image_path)
        else:
            negative_images.append(image_path)

    randomizer = random.Random(seed)
    randomizer.shuffle(positive_images)
    randomizer.shuffle(negative_images)

    def _split_group(items: list[Path]) -> tuple[list[Path], list[Path]]:
        if len(items) <= 1:
            return items[:], []

        train_count = max(1, int(round(len(items) * train_ratio)))
        train_count = min(train_count, len(items) - 1)
        return items[:train_count], items[train_count:]

    train_positive, val_positive = _split_group(positive_images)
    train_negative, val_negative = _split_group(negative_images)
    train_images = sorted(train_positive + train_negative)
    val_images = sorted(val_positive + val_negative)

    for split_name, split_images in (("train", train_images), ("val", val_images)):
        (output_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            label_path = dataset.labels_dir / f"{image_path.stem}.txt"
            shutil.copy2(image_path, output_root / split_name / "images" / image_path.name)
            shutil.copy2(label_path, output_root / split_name / "labels" / label_path.name)

    shutil.copy2(dataset.classes_path, output_root / "classes.txt")
    if dataset.notes_path is not None:
        shutil.copy2(dataset.notes_path, output_root / "notes.json")

    class_names = load_class_names(dataset)
    dataset_yaml_path = output_root / "dataset.yaml"
    dataset_yaml_path.write_text(
        "\n".join(
            [
                f"path: '{output_root.as_posix()}'",
                "train: train/images",
                "val: val/images",
                "",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(class_names)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "source_root": str(dataset.root),
        "output_root": str(output_root),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_positive_images": len(train_positive),
        "train_negative_images": len(train_negative),
        "val_positive_images": len(val_positive),
        "val_negative_images": len(val_negative),
        "dataset_yaml": str(dataset_yaml_path),
    }
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
