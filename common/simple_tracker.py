from __future__ import annotations

import math


class SimpleTracker:
    def __init__(self, max_distance: float = 120.0, max_missed_frames: int = 10) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_track_id = 1
        self.tracks: dict[int, dict[str, float | int]] = {}

    def update(self, detections: list[dict[str, float | int]]) -> list[dict[str, float | int | bool]]:
        assignments: list[dict[str, float | int | bool]] = []
        assigned_detection_indexes: set[int] = set()
        unmatched_track_ids = set(self.tracks)

        pairings: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                distance = math.hypot(
                    float(detection["center_x"]) - float(track["center_x"]),
                    float(detection["center_y"]) - float(track["center_y"]),
                )
                pairings.append((distance, track_id, detection_index))

        matched_track_ids: set[int] = set()
        for distance, track_id, detection_index in sorted(pairings, key=lambda item: item[0]):
            if distance > self.max_distance:
                continue
            if track_id in matched_track_ids or detection_index in assigned_detection_indexes:
                continue

            track = self.tracks[track_id]
            detection = dict(detections[detection_index])
            detection["track_id"] = track_id
            detection["previous_center_x"] = float(track["center_x"])
            detection["previous_center_y"] = float(track["center_y"])
            detection["is_new"] = False
            assignments.append(detection)

            self.tracks[track_id] = {
                "center_x": float(detection["center_x"]),
                "center_y": float(detection["center_y"]),
                "width": float(detection["width"]),
                "height": float(detection["height"]),
                "confidence": float(detection["confidence"]),
                "missed_frames": 0,
            }

            matched_track_ids.add(track_id)
            assigned_detection_indexes.add(detection_index)
            unmatched_track_ids.discard(track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detection_indexes:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            assigned_detection = dict(detection)
            assigned_detection["track_id"] = track_id
            assigned_detection["previous_center_x"] = float(detection["center_x"])
            assigned_detection["previous_center_y"] = float(detection["center_y"])
            assigned_detection["is_new"] = True
            assignments.append(assigned_detection)

            self.tracks[track_id] = {
                "center_x": float(detection["center_x"]),
                "center_y": float(detection["center_y"]),
                "width": float(detection["width"]),
                "height": float(detection["height"]),
                "confidence": float(detection["confidence"]),
                "missed_frames": 0,
            }

        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            missed_frames = int(track["missed_frames"]) + 1
            if missed_frames > self.max_missed_frames:
                del self.tracks[track_id]
            else:
                track["missed_frames"] = missed_frames

        assignments.sort(key=lambda item: int(item["track_id"]))
        return assignments
