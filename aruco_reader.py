#!/usr/bin/env python3
"""
ArUco reader for a wired (USB) camera.

Your marker set: 50 codes, dictionary DICT_5X5_1000, IDs 1-50 (from the "Mã ArUco" folder).
For every code seen by the camera the program returns: ID, centre (px), 4 corners (px),
rotation angle (deg) and an optional label.

Install:   pip install opencv-contrib-python numpy

Usage
  python aruco_reader.py --list-cameras              # find which camera index works
  python aruco_reader.py                             # camera 0, print events, no window
  python aruco_reader.py --show                      # + live window (press q to quit)
  python aruco_reader.py --source 1 --width 1280 --height 720 --show
  python aruco_reader.py --json --log scans.csv      # JSON lines on stdout + CSV log
  python aruco_reader.py --every-frame               # report every detection in every frame
  python aruco_reader.py --source photo.jpg --out annotated.jpg
  python aruco_reader.py --source clip.mp4
  python aruco_reader.py --labels labels.json        # {"1": "Station A", "2": "Station B"}
  python aruco_reader.py --selftest "Mã ArUco"       # check the 50 SVG files decode correctly

Use from your own code
  from aruco_reader import ArucoReader
  reader = ArucoReader()                 # DICT_5X5_1000, IDs 1-50
  for r in reader.read(frame):           # frame = BGR image from cv2
      print(r.id, r.center, r.angle_deg)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import cv2
import numpy as np

DEFAULT_DICT = "DICT_5X5_1000"
DEFAULT_IDS = "1-50"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- core
@dataclass
class ArucoResult:
    id: int
    center: tuple          # (x, y) in pixels
    corners: list          # 4 x [x, y]: marker's own top-left, top-right, bottom-right, bottom-left
    angle_deg: float       # rotation of the marker's top edge vs. image x-axis
    label: str | None = None

    def to_dict(self):
        return asdict(self)


def parse_ids(spec):
    """'1-50' / '1,5,10-20' -> set of ints.  'all' -> None (accept any ID in the dictionary)."""
    if spec is None or spec.strip().lower() == "all":
        return None
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        elif part:
            ids.add(int(part))
    return ids


class ArucoReader:
    def __init__(self, dict_name=DEFAULT_DICT, allowed_ids=DEFAULT_IDS, labels=None):
        """allowed_ids: IDs you actually printed. Anything else is treated as a false detection."""
        self.allowed = parse_ids(allowed_ids) if isinstance(allowed_ids, str) else (
            None if allowed_ids is None else set(allowed_ids))
        self.labels = {int(k): v for k, v in (labels or {}).items()}
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detect = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers
        else:  # older OpenCV
            params = cv2.aruco.DetectorParameters_create()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detect = lambda g: cv2.aruco.detectMarkers(g, dictionary, parameters=params)

    def read(self, frame) -> list:
        """Decode every ArUco code in a frame. Returns a list of ArucoResult sorted by ID."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        results = []
        if ids is None:
            return results
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if self.allowed is not None and i not in self.allowed:
                continue
            pts = c.reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            dx, dy = pts[1] - pts[0]
            results.append(ArucoResult(
                id=i,
                center=(round(float(cx), 1), round(float(cy), 1)),
                corners=[[round(float(x), 1), round(float(y), 1)] for x, y in pts],
                angle_deg=round(math.degrees(math.atan2(dy, dx)), 1),
                label=self.labels.get(i),
            ))
        results.sort(key=lambda r: r.id)
        return results

    @staticmethod
    def draw(frame, results):
        for r in results:
            pts = np.array(r.corners, np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.circle(frame, tuple(pts[0]), 5, (0, 0, 255), -1)  # marks the marker's top-left corner
            text = f"ID {r.id}" + (f" {r.label}" if r.label else "")
            cv2.putText(frame, text, (int(r.center[0]) - 30, int(r.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, text, (int(r.center[0]) - 30, int(r.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame


class Tracker:
    """Turns noisy per-frame detections into stable events.
    A code is 'found' after `confirm` consecutive frames and 'lost' after `lost` frames without it."""

    def __init__(self, confirm=3, lost=15):
        self.confirm, self.lost = confirm, lost
        self.state = {}  # id -> dict(streak, missing, active)

    def update(self, results):
        events, now = [], {r.id: r for r in results}
        for i, r in now.items():
            s = self.state.setdefault(i, {"streak": 0, "missing": 0, "active": False})
            s["streak"] += 1
            s["missing"] = 0
            if not s["active"] and s["streak"] >= self.confirm:
                s["active"] = True
                events.append(("found", r))
        for i, s in list(self.state.items()):
            if i in now:
                continue
            s["streak"] = 0
            s["missing"] += 1
            if s["active"] and s["missing"] >= self.lost:
                s["active"] = False
                events.append(("lost", i))
            if not s["active"] and s["missing"] >= self.lost:
                del self.state[i]
        return events


# ---------------------------------------------------------------------- camera
def open_camera(index, width=None, height=None, mjpg=False):
    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow opens USB cameras faster on Windows
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # smaller buffer = less lag (ignored by some drivers)
    return cap


def list_cameras(max_index=6):
    found = False
    for i in range(max_index):
        cap = open_camera(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                print(f"camera {i}: {frame.shape[1]}x{frame.shape[0]}")
                found = True
        cap.release()
    if not found:
        print("No camera found. Check the cable / permissions, or try another USB port.")


# ---------------------------------------------------------------------- output
def now_str():
    return datetime.now().isoformat(timespec="milliseconds")


class Emitter:
    def __init__(self, as_json=False, log_path=None):
        self.as_json = as_json
        self.csv_file = self.csv = None
        if log_path:
            new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
            self.csv_file = open(log_path, "a", newline="", encoding="utf-8")
            self.csv = csv.writer(self.csv_file)
            if new:
                self.csv.writerow(["time", "event", "id", "label", "cx", "cy", "angle_deg"])

    def emit(self, kind, item, labels=None):
        t = now_str()
        if kind == "lost":
            rid, label, cx, cy, ang = item, (labels or {}).get(item), "", "", ""
            payload = {"time": t, "event": "lost", "id": rid, "label": label}
            line = f"[{t}] LOST  id={rid}" + (f" ({label})" if label else "")
        else:
            r = item
            rid, label, (cx, cy), ang = r.id, r.label, r.center, r.angle_deg
            payload = {"time": t, "event": kind, **r.to_dict()}
            line = (f"[{t}] {kind.upper():5} id={r.id}" + (f" ({r.label})" if r.label else "")
                    + f" center=({cx:.0f},{cy:.0f}) angle={ang:.0f}deg")
        print(json.dumps(payload, ensure_ascii=False) if self.as_json else line, flush=True)
        if self.csv:
            self.csv.writerow([t, kind, rid, label or "", cx, cy, ang])
            self.csv_file.flush()

    def close(self):
        if self.csv_file:
            self.csv_file.close()


# ---------------------------------------------------------------------- modes
def run_image(args, reader, emitter):
    frame = cv2.imread(args.source)
    if frame is None:
        sys.exit(f"Cannot read image: {args.source}")
    results = reader.read(frame)
    for r in results:
        emitter.emit("found", r)
    print(f"{len(results)} code(s) found", file=sys.stderr)
    if args.out or args.show:
        ArucoReader.draw(frame, results)
    if args.out:
        cv2.imwrite(args.out, frame)
    if args.show:
        cv2.imshow("aruco", frame)
        cv2.waitKey(0)


def run_stream(args, reader, emitter):
    is_cam = args.source.isdigit()
    cap = open_camera(int(args.source), args.width, args.height, args.mjpg) if is_cam \
        else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Cannot open source '{args.source}'. Run with --list-cameras to find the right index.")
    tracker = Tracker(args.confirm, args.lost)
    writer, last_t, fps = None, time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_cam:
                    sys.exit("Camera stopped sending frames (cable unplugged?)")
                break  # end of video file
            results = reader.read(frame)
            if args.every_frame:
                for r in results:
                    emitter.emit("seen", r)
            else:
                for kind, item in tracker.update(results):
                    emitter.emit(kind, item, reader.labels)
            if args.show or args.out:
                ArucoReader.draw(frame, results)
                fps = 0.9 * fps + 0.1 / max(time.time() - last_t, 1e-6)
                last_t = time.time()
                cv2.putText(frame, f"{fps:.0f} fps | {len(results)} code(s)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                             cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
                writer.write(frame)
            if args.show:
                cv2.imshow("ArUco reader (q = quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


def selftest(folder, reader):
    """Rasterise the provided SVG files and confirm each decodes to the ID in its filename."""
    files = glob.glob(os.path.join(folder, "**", "*.svg"), recursive=True)
    if not files:
        sys.exit(f"No .svg files under {folder}")
    rect_re = re.compile(r"<rect([^>]*)>")
    ok = 0
    for path in sorted(files, key=lambda p: int(re.search(r"-(\d+)\.svg$", p).group(1))):
        want = int(re.search(r"-(\d+)\.svg$", path).group(1))
        grid = np.zeros((7, 7), np.uint8)  # 5x5 bits + 1-cell black border
        for m in rect_re.finditer(open(path, encoding="utf-8").read()):
            a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            if a.get("fill") == "white":  # a white rect's origin = a white cell (1.5 sizes only hide seams)
                grid[int(float(a["y"])), int(float(a["x"]))] = 255
        img = cv2.resize(grid, None, fx=60, fy=60, interpolation=cv2.INTER_NEAREST)
        img = cv2.copyMakeBorder(img, 120, 120, 120, 120, cv2.BORDER_CONSTANT, value=255)
        got = [r.id for r in reader.read(img)]
        status = "OK " if got == [want] else "BAD"
        ok += got == [want]
        if got != [want]:
            print(f"{status} {os.path.basename(path)} -> {got}")
    print(f"{ok}/{len(files)} markers decode to the ID in their filename")


# ---------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index (default 0), video file, or image file")
    p.add_argument("--dict", default=DEFAULT_DICT, help=f"ArUco dictionary (default {DEFAULT_DICT})")
    p.add_argument("--ids", default=DEFAULT_IDS, help="IDs to accept, e.g. 1-50 or 1,5,9-12 or all")
    p.add_argument("--labels", help="JSON file mapping ID -> label/meaning")
    p.add_argument("--width", type=int, help="camera width, e.g. 1280")
    p.add_argument("--height", type=int, help="camera height, e.g. 720")
    p.add_argument("--mjpg", action="store_true", help="request MJPG from the camera (higher FPS on many USB cams)")
    p.add_argument("--confirm", type=int, default=3, help="frames a code must be seen before it is reported")
    p.add_argument("--lost", type=int, default=15, help="frames a code must be missing before it is 'lost'")
    p.add_argument("--every-frame", action="store_true", help="report every detection in every frame")
    p.add_argument("--json", action="store_true", help="print results as JSON lines")
    p.add_argument("--log", help="append results to this CSV file")
    p.add_argument("--show", action="store_true", help="show live window")
    p.add_argument("--out", help="save annotated image / video")
    p.add_argument("--list-cameras", action="store_true", help="probe camera indexes 0-5 and exit")
    p.add_argument("--selftest", metavar="SVG_FOLDER", help="verify the provided SVG markers and exit")
    args = p.parse_args()

    if args.list_cameras:
        return list_cameras()
    labels = None
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            labels = json.load(f)
    reader = ArucoReader(args.dict, args.ids, labels)
    if args.selftest:
        return selftest(args.selftest, ArucoReader(args.dict, "all"))

    emitter = Emitter(args.json, args.log)
    try:
        if os.path.splitext(args.source)[1].lower() in IMAGE_EXTS:
            run_image(args, reader, emitter)
        else:
            run_stream(args, reader, emitter)
    except BrokenPipeError:
        pass
    finally:
        emitter.close()


if __name__ == "__main__":
    main()
