
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-object local-template monitor
- Matches multiple templates (USB + GripTok + any others) in a single camera stream.
- Flags "pattern deviation" when position or appearance leaves allowed bounds with hysteresis + grace.
- Designed to be EASY to extend: just edit OBJECTS below, or drop images into AUTOLOAD_DIR.

Controls:
  ESC : quit
"""

import cv2
import numpy as np
import time
import glob
import os
from typing import Tuple, Dict, Any, List

# =========================
# User configuration
# =========================
SOURCE = 1  # camera index or video file path

# Option A) Explicit list of tracked objects (recommended for clarity)
OBJECTS: List[Dict[str, Any]] = [
    {"name": "usb",     "template_path": "../../assets/usb_test2.jpg",    "color": (0, 255, 0)},
    {"name": "griptok", "template_path": "../../assets/griptok_test.jpg", "color": (255, 128, 0)},
]

# Option B) Autoload all templates in a folder (png/jpg). If not used, set to "".
# Files will be tracked by their filename (without extension).
AUTOLOAD_DIR = "../../assets/"  # e.g., "assets/templates_autoload"

# Frame & preprocessing
FRAME_MAX_W = 0
STAB_SCALE = 0.0  # 0 = off; try 0.5~0.7 if the camera shakes a lot
BLUR_K = 3

# Template rotations to test (degrees). Use [0] if rotation is fixed.
ROT_DEGS = [0, -10, +10, -20, +20]

# Matching
METHOD = cv2.TM_CCOEFF_NORMED
MIN_TRUST_SCORE = 0.35  # minimum template confidence to accept new match

# Windows/patches
ALLOWED_BOX_SCALE = 1.0   # allowed deviation box = tw0/th0 * this scale around HOME
SEARCH_WIN_SCALE = 2.2    # local search window scale
PATCH_SCALE = 1.2         # appearance patch scale (relative to template w/h)

# Motion gate (optional): ignore position deviation while motion in allowed/home box is large
MOTION_GATE = True
MORPH_K = 5
MOTION_AREA_THRESH = 0.05  # fraction of foreground pixels in allowed box to consider "occluded"

# Hysteresis & timing
EMA_ALPHA = 0.35           # Exponential moving average weight for center smoothing
DEVIATE_FRAMES_REQ = 3     # number of consecutive deviated frames to trigger strong_deviate
APPEAR_FRAMES_REQ = 3
GRACE_SECS = 2.0           # time to return before alert

ANALYZE_EVERY = 1          # process every Nth frame (for CPU saving)

WINDOW_NAME = "multi_template_monitor"


# =========================
# Utilities
# =========================
def resize_keep_w(img, max_w: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    scale = max_w / float(w)
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out, scale


def preprocess_for_matching(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    if BLUR_K and BLUR_K >= 3:
        gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    return gray


def rotate_img(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def best_match_global(search_img_gray: np.ndarray,
                      tgray_rots: List[Tuple[float, np.ndarray]]) -> Dict[str, Any]:
    best = {"score": -1.0, "top_left": (0, 0), "size": (None, None), "deg": 0}
    H, W = search_img_gray.shape[:2]
    for deg, tgray in tgray_rots:
        th, tw = tgray.shape[:2]
        if tw is None or th is None or tw < 8 or th < 8 or tw > W or th > H:
            continue
        res = cv2.matchTemplate(search_img_gray, tgray, METHOD)
        minv, maxv, minl, maxl = cv2.minMaxLoc(res)
        score = maxv if METHOD in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else (1.0 - minv)
        if score > best["score"]:
            best.update({"score": score, "top_left": maxl, "size": (tw, th), "deg": deg})
    return best


def center_from(top_left: Tuple[int, int], size: Tuple[int, int]) -> Tuple[int, int]:
    x, y = top_left
    tw, th = size
    return (int(x + tw / 2), int(y + th / 2))


def clamp_box(cx: int, cy: int, hw: int, hh: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, cx - hw); y1 = max(0, cy - hh)
    x2 = min(W, cx + hw); y2 = min(H, cy + hh)
    return x1, y1, x2, y2


def crop_patch(gray: np.ndarray, center: Tuple[int, int], size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    cx, cy = center
    w, h = size
    W, H = gray.shape[1], gray.shape[0]
    x1, y1, x2, y2 = clamp_box(cx, cy, w // 2, h // 2, W, H)
    patch = gray[y1:y2, x1:x2]
    return patch, (x1, y1, x2, y2)


def stabilize_ecc(prev_small: np.ndarray, curr_small: np.ndarray) -> np.ndarray:
    # Rigid transform (rotation+translation) approx with ECC
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        cc, warp_matrix = cv2.findTransformECC(prev_small, curr_small, warp_matrix, warp_mode, criteria, None, 1)
    except cv2.error:
        pass
    return warp_matrix


# =========================
# Object loader
# =========================
def load_objects() -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []

    # A) Explicit list
    for cfg in OBJECTS:
        path = cfg["template_path"]
        timg = cv2.imread(path)
        if timg is None:
            print(f"[WARN] Cannot read template: {path}")
            continue
        timg, _ = resize_keep_w(timg, 300)
        tgray_rots = [(deg, preprocess_for_matching(rotate_img(timg, deg))) for deg in ROT_DEGS]
        objs.append({
            "name": cfg["name"],
            "color": cfg.get("color", (0, 255, 0)),
            "template_img": timg,
            "templates_gray": tgray_rots,
            "home": None,
            "home_patch": None,
            "tw0": None,
            "th0": None,
            "ema_center": None,
            "deviate_since": None,
            "deviate_run": 0,
            "appear_run": 0
        })

    # B) Autoload folder (optional). Skips duplicates by name.
    if AUTOLOAD_DIR and os.path.isdir(AUTOLOAD_DIR):
        for fp in sorted(glob.glob(os.path.join(AUTOLOAD_DIR, "*.*"))):
            if not fp.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue
            name = os.path.splitext(os.path.basename(fp))[0]
            if any(o["name"] == name for o in objs):
                continue
            timg = cv2.imread(fp)
            if timg is None:
                print(f"[WARN] Cannot read template: {fp}")
                continue
            timg, _ = resize_keep_w(timg, 300)
            tgray_rots = [(deg, preprocess_for_matching(rotate_img(timg, deg))) for deg in ROT_DEGS]
            objs.append({
                "name": name,
                "color": (128, 255, 0),
                "template_img": timg,
                "templates_gray": tgray_rots,
                "home": None,
                "home_patch": None,
                "tw0": None,
                "th0": None,
                "ema_center": None,
                "deviate_since": None,
                "deviate_run": 0,
                "appear_run": 0
            })

    return objs


# =========================
# Main
# =========================
def main():
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {SOURCE}")

    objects = load_objects()
    if not objects:
        raise SystemExit("No templates loaded. Check OBJECTS or AUTOLOAD_DIR.")

    # First frame
    ok, first = cap.read()
    if not ok:
        raise SystemExit("Cannot read first frame.")
    first, _ = resize_keep_w(first, FRAME_MAX_W)
    first_fg = preprocess_for_matching(first)

    # Initialize homes per object
    for obj in objects:
        bm = best_match_global(first_fg, obj["templates_gray"])
        top_left, score, (tw, th) = bm["top_left"], bm["score"], bm["size"]
        if tw is None or th is None or tw < 8 or th < 8 or score < 0.1:
            raise SystemExit(f"[{obj['name']}] template not confidently found in first frame. Score={score:.2f}")
        home = center_from(top_left, (tw, th))
        obj["home"] = home
        obj["ema_center"] = home
        obj["tw0"], obj["th0"] = tw, th

        # appearance/home patch
        home_patch, _ = crop_patch(first_fg, home, (int(PATCH_SCALE * tw), int(PATCH_SCALE * th)))
        if home_patch.size == 0:
            raise SystemExit(f"[{obj['name']}] home_patch crop failed — reduce PATCH_SCALE.")
        obj["home_patch"] = home_patch

    # Background subtractor & morphology (shared)
    bg = None
    kernel = None
    if MOTION_GATE:
        bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))

    prev_small = None
    fcount = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        fcount += 1
        if fcount % max(1, ANALYZE_EVERY) != 0:
            continue

        frame, _ = resize_keep_w(frame, FRAME_MAX_W)
        fg = preprocess_for_matching(frame)

        # Optional ECC stabilization
        if STAB_SCALE and STAB_SCALE > 0:
            small = cv2.resize(fg, None, fx=STAB_SCALE, fy=STAB_SCALE, interpolation=cv2.INTER_AREA)
            if prev_small is not None:
                warp = stabilize_ecc(prev_small, small)
                fg = cv2.warpAffine(fg, warp, (fg.shape[1], fg.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                frame = cv2.warpAffine(frame, warp, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            prev_small = small

        # Motion gate mask (shared)
        m = None
        if MOTION_GATE and bg is not None:
            m = bg.apply(frame)
            _, m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

        vis = frame.copy()
        global_alert = False

        H, W = fg.shape[:2]

        for idx, obj in enumerate(objects):
            name = obj["name"]
            color = obj["color"]
            home = obj["home"]
            ema_center = obj["ema_center"]
            tw0, th0 = obj["tw0"], obj["th0"]
            home_patch = obj["home_patch"]

            # Local search window around EMA
            allowed_half_w = int(ALLOWED_BOX_SCALE * tw0)
            allowed_half_h = int(ALLOWED_BOX_SCALE * th0)
            search_half_w = int(SEARCH_WIN_SCALE * tw0)
            search_half_h = int(SEARCH_WIN_SCALE * th0)

            sx1, sy1, sx2, sy2 = clamp_box(ema_center[0], ema_center[1], search_half_w, search_half_h, W, H)
            local = fg[sy1:sy2, sx1:sx2]

            bm = best_match_global(local, obj["templates_gray"])
            (lx, ly) = bm["top_left"]
            top_left = (sx1 + lx, sy1 + ly)
            score = bm["score"]
            tw, th = bm["size"]

            if score < MIN_TRUST_SCORE or tw is None or th is None:
                match_center = ema_center
            else:
                match_center = center_from(top_left, (tw, th))

            # EMA update
            ema_center = (
                int(EMA_ALPHA * match_center[0] + (1 - EMA_ALPHA) * ema_center[0]),
                int(EMA_ALPHA * match_center[1] + (1 - EMA_ALPHA) * ema_center[1])
            )
            obj["ema_center"] = ema_center

            # Occlusion gate via motion ratio inside allowed box around HOME
            occluded = False
            x1a, y1a, x2a, y2a = clamp_box(home[0], home[1], allowed_half_w, allowed_half_h, W, H)
            if MOTION_GATE and m is not None:
                m_crop = m[y1a:y2a, x1a:x2a]
                move_ratio = float(m_crop.sum()) / (255.0 * max(1, m_crop.size))
                occluded = (move_ratio >= MOTION_AREA_THRESH)

            # Position deviation (if not occluded)
            if occluded:
                pos_deviated = False
            else:
                pos_deviated = not (x1a <= ema_center[0] <= x2a and y1a <= ema_center[1] <= y2a)

            # Appearance deviation
            curr_patch, curr_box = crop_patch(fg, ema_center, (int(PATCH_SCALE * tw0), int(PATCH_SCALE * th0)))
            app_score = 1.0
            app_deviated = False
            if curr_patch.size and home_patch.size and (curr_patch.shape[0] >= 8 and curr_patch.shape[1] >= 8):
                res = cv2.matchTemplate(curr_patch, home_patch, METHOD)
                minv, maxv, _, _ = cv2.minMaxLoc(res)
                app_score = maxv if METHOD in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else (1.0 - minv)
                app_deviated = (app_score < 0.55)  # default threshold for appearance difference

            # Hysteresis counters
            deviated_now = pos_deviated or app_deviated
            obj["deviate_run"] = obj["deviate_run"] + 1 if deviated_now else 0
            obj["appear_run"] = obj["appear_run"] + 1 if app_deviated else 0

            strong_deviate = (obj["deviate_run"] >= DEVIATE_FRAMES_REQ) or (obj["appear_run"] >= APPEAR_FRAMES_REQ)

            alert = False
            now = time.time()
            if strong_deviate:
                if obj["deviate_since"] is None:
                    obj["deviate_since"] = now
            else:
                obj["deviate_since"] = None

            remain = None
            if obj["deviate_since"] is not None:
                elapsed = now - obj["deviate_since"]
                remain = max(0, int(GRACE_SECS - elapsed))
                if elapsed >= GRACE_SECS:
                    alert = True
                    global_alert = True

            # Visuals (per object)
            if tw and th and tw > 0 and th > 0:
                cv2.rectangle(vis, top_left, (top_left[0] + tw, top_left[1] + th), color, 2)
            cv2.rectangle(vis, (x1a, y1a), (x2a, y2a), (0, 255, 255), 2)
            cv2.circle(vis, home, 5, (255, 255, 0), -1)
            cv2.circle(vis, ema_center, 5, (0, 0, 255) if deviated_now else (0, 255, 0), -1)
            cv2.line(vis, home, ema_center, (0, 0, 255) if deviated_now else (0, 255, 0), 2)

            ybase = 30 + idx * 50
            cv2.putText(vis, f"[{name}] score={score:.2f} app={app_score:.2f} occ={int(occluded)}",
                        (20, ybase), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if obj["deviate_since"] is not None and not alert and remain is not None:
                cv2.putText(vis, f"[{name}] Return within {remain}s",
                            (20, ybase + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if alert:
                cv2.putText(vis, f"[{name}] ALERT: Pattern deviated!", (20, ybase + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if global_alert:
            cv2.putText(vis, "GLOBAL ALERT: Some pattern(s) deviated!",
                        (20, 15 + len(objects)*50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.imshow(WINDOW_NAME, vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
