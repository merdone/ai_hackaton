import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from settings import WorkerSettings, get_settings


@dataclass
class Zone:
    name: str
    points: list[tuple[int, int]]


class ZoneAnnotator:
    def __init__(self, settings: WorkerSettings, frame: np.ndarray, output_path: Path) -> None:
        self.base_frame = frame
        self.output_path = output_path
        self.zone_annotator_window_name = settings.zone_annotator_window_name
        self.current_points: list[tuple[int, int]] = []
        self.zones: list[Zone] = []

        cv2.namedWindow(self.zone_annotator_window_name)
        cv2.setMouseCallback(self.zone_annotator_window_name, self.on_mouse)

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))

    def draw(self) -> np.ndarray:
        canvas = self.base_frame.copy()

        for zone in self.zones:
            pts = np.array(zone.points, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(
                canvas,
                zone.name,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(255, 0, 255), thickness=2)
            for i, (x, y) in enumerate(self.current_points):
                cv2.circle(canvas, (x, y), 4, (255, 255, 0), -1)
                cv2.putText(
                    canvas,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        help_text = "LMB:add | z:undo | c:close zone | r:reset draft | s:save | q:quit"
        cv2.putText(
            canvas,
            help_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def close_zone(self) -> None:
        if len(self.current_points) < 3:
            print("Need at least 3 points to close a polygon.")
            return

        zone_name = input("Zone name: ").strip()
        if not zone_name:
            zone_name = f"Zone_{len(self.zones) + 1}"

        self.zones.append(Zone(name=zone_name, points=list(self.current_points)))
        self.current_points = []
        print(f"Saved polygon: {zone_name}")

    def save(self) -> None:
        payload = {
            "image_width": int(self.base_frame.shape[1]),
            "image_height": int(self.base_frame.shape[0]),
            "zones": [
                {
                    "name": zone.name,
                    "points": [[int(x), int(y)] for x, y in zone.points],
                }
                for zone in self.zones
            ],
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Exported: {self.output_path}")

    def run(self) -> None:
        while True:
            canvas = self.draw()
            cv2.imshow(self.zone_annotator_window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break
            if key == ord("z") and self.current_points:
                self.current_points.pop()
            if key == ord("r"):
                self.current_points = []
            if key == ord("c"):
                self.close_zone()
            if key == ord("s"):
                self.save()

        cv2.destroyAllWindows()


def load_frame(settings: WorkerSettings, image_path: str | None, video_path: str | None) -> np.ndarray:
    if image_path:
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (settings.yolo_preview_width, settings.yolo_preview_height),)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        return frame

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        ok, frame = cap.read()
        cap.release()
        frame = cv2.resize(frame, (settings.yolo_preview_width, settings.yolo_preview_height), )
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read first frame from: {video_path}")
        return frame

    raise RuntimeError("Provide --image or --video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polygon zone annotator")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Path to image")
    source_group.add_argument("--video", type=str, help="Path to video (first frame is used)")
    parser.add_argument("--out", type=str, default="../data/output/zones.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    frame = load_frame(settings, args.image, args.video)
    annotator = ZoneAnnotator(settings, frame=frame, output_path=Path(args.out))
    annotator.run()


if __name__ == "__main__":
    main()
