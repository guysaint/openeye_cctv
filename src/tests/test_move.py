#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2b — 엣지 기반 템플릿 매칭 + FPS 스케줄러 (LOCK 후 변화 감지)
- 첫 단계: 템플릿을 전체 프레임에서 매칭해 위치 탐색(SEARCH)
- 이후: 해당 위치를 고정(LOCKED)하고, 같은 ROI에서 '변화'만 감지
- 변화 감지: 엣지(프레임) vs 엣지(템플릿) 평균 차이(정규화) + 연속 프레임 히스테리시스
"""

import cv2
import numpy as np
import argparse
from typing import Tuple
import time

# ---------------- 유틸 ----------------
def resize_keep_w(img, max_w: int) -> Tuple:
    """가로폭 기준 리사이즈 (표시/속도 조절). (resized, scale) 반환"""
    h, w = img.shape[:2]
    if max_w and w > max_w:
        scale = max_w / float(w)
        return cv2.resize(img, (int(w * scale), int(h * scale))), scale
    return img, 1.0

def get_method(name: str) -> int:
    table = {
        "TM_SQDIFF": cv2.TM_SQDIFF,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    }
    return table.get(name.upper(), cv2.TM_CCORR)  # 안전하게 CCORR 기본

def edges_clean(
    bgr,
    blur=3,
    k1=0.66,     # median 기반 아래 임계 비율
    k2=1.33,     # median 기반 위 임계 비율
    op: str = "none",   # "opening" | "closing" | "both" | "none"
    ksize: int = 3,
    open_iter: int = 1,
    close_iter: int = 1,
    min_area: int = 0   # 0이면 면적 필터 끔
):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur:
        g = cv2.GaussianBlur(g, (blur, blur), 0)

    med = float(np.median(g))
    lo = int(max(0, (1.0 - k1) * med))
    hi = int(min(255, (1.0 + k2) * med))
    e = cv2.Canny(g, lo, hi)

    if op != "none":
        K = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        if op in ("opening", "both"):
            e = cv2.morphologyEx(e, cv2.MORPH_OPEN,  K, iterations=open_iter)
        if op in ("closing", "both"):
            e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, K, iterations=close_iter)
            if ksize >= 3:
                e = cv2.erode(e, K, iterations=1)

    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((e > 0).astype(np.uint8), connectivity=8)
        mask = np.zeros_like(e)
        for i in range(1, n):  # 0 = background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255
        e = mask

    return e

# ---------------- 메인 ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Edge-based template matching with FPS scheduler (lock & change detection)")
    p.add_argument("--video", type=str, default="../../assets/chair_test_480p_30fps.mp4",
                   help="영상 경로 또는 0(웹캠). 기본: ../../assets/chair_test_480p_30fps.mp4")
    p.add_argument("--template", type=str, default="../../assets/chair_template.png",
                   help="템플릿 이미지 경로 (의자 ROI)")
    p.add_argument("--maxw", type=int, default=1280,
                   help="프레임 표시/처리용 최대 가로폭 (속도/가독성). 기본: 1280")
    p.add_argument("--method", type=str, default="TM_CCORR",
                   help="매칭 방법. 엣지 기반에선 TM_CCORR 권장(정규화 문제 회피)")
    # 형태학 파라미터(필요시 CLI로 튜닝)
    p.add_argument("--tmpl-op", type=str, default="closing",
                   help="템플릿: none|opening|closing|both (기본: closing)")
    p.add_argument("--tmpl-min-area", type=int, default=10,
                   help="템플릿: 작은 조각 제거 임계(픽셀). 기본: 10")
    p.add_argument("--frm-op", type=str, default="none",
                   help="프레임: none|opening|closing|both (기본: none)")
    p.add_argument("--frm-min-area", type=int, default=0,
                   help="프레임: 작은 조각 제거 임계(픽셀). 기본: 0")

    # 변화 감지 파라미터
    p.add_argument("--diff-th", type=float, default=0.12,
                   help="변화 감지 임계 (0~1). 값↑=더 엄격")
    p.add_argument("--persist", type=int, default=6,
                   help="임계 이상이 연속 N프레임일 때 경고")
    return p.parse_args()

def main():
    args = parse_args()

    # 입력/템플릿
    source = 0 if str(args.video) == "0" else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[Error] Cannot open source: {args.video}")

    tmpl_bgr = cv2.imread(args.template)
    if tmpl_bgr is None:
        cap.release()
        raise SystemExit(f"[Error] Cannot read template: {args.template}")

    # 템플릿 엣지 & 마스크
    tmpl_edges = edges_clean(
        tmpl_bgr, op=args.tmpl_op, ksize=3, open_iter=1, close_iter=1, min_area=args.tmpl_min_area
    )
    tmpl_mask = (tmpl_edges > 0).astype(np.uint8)
    th, tw = tmpl_edges.shape[:2]
    print(f"[Info] Template edge size: {tw}x{th}")

    method = get_method(args.method)
    print(f"[Info] Matching method: {args.method}")

    # FPS 스케줄러
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    frame_period = 1.0 / fps
    next_tick = time.perf_counter() + frame_period
    DROP_LATE_FRAMES = True
    delay_ms = int(frame_period * 1000)
    print(f"[Info] Source FPS: {fps:.2f} (period={delay_ms} ms)")

    # 상태: SEARCH(위치 찾기) → LOCKED(고정 ROI 변화 감지)
    mode = "SEARCH"
    lock_xy = None           # (x1, y1)
    alert_count = 0          # 연속 프레임 카운터
    last_diff = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_resized, _ = resize_keep_w(frame, args.maxw)
        frm_edges = edges_clean(
            frame_resized, op=args.frm_op, ksize=3, open_iter=1, close_iter=1, min_area=args.frm_min_area
        )
        H, W = frm_edges.shape[:2]

        if th > H or tw > W:
            vis = frame_resized.copy()
            cv2.putText(vis, "Template larger than frame — use smaller template.",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            if mode == "SEARCH":
                # --- 전역 매칭으로 최초 위치 탐색 ---
                res = cv2.matchTemplate(frm_edges, tmpl_edges, method, mask=tmpl_mask)
                res = np.nan_to_num(res, nan=-1.0, posinf=1e9, neginf=-1e9)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    top_left = min_loc
                    score = 1.0 - float(min_val)
                else:
                    top_left = max_loc
                    score = float(max_val)

                x1, y1 = top_left
                x2, y2 = x1 + tw, y1 + th

                # 위치를 잠궈도 좋을 만큼 매칭이 나왔는지(느슨히 0.4~0.6 사이로 조정 가능)
                if score >= 0.5:
                    lock_xy = (x1, y1)
                    mode = "LOCKED"

                vis = frame_resized.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"[SEARCH] score={score:.3f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                # --- LOCKED: 고정 ROI에서 '변화'만 감지 ---
                x1, y1 = lock_xy
                x2, y2 = x1 + tw, y1 + th
                # 경계 보정
                x1 = max(0, min(W - tw, x1)); x2 = x1 + tw
                y1 = max(0, min(H - th, y1)); y2 = y1 + th

                roi_edges = frm_edges[y1:y2, x1:x2]
                # 템플릿 엣지와의 차이(0~255) → 평균 → 0~1로 정규화
                # 엣지는 0/255 값 비중이 커서 mean(absdiff)/255가 간단하고 강건함
                diff = float(cv2.mean(cv2.absdiff(roi_edges, tmpl_edges))[0]) / 255.0
                last_diff = diff

                # 히스테리시스: 연속 persist 프레임 이상이면 경고
                if diff >= args.diff_th:
                    alert_count += 1
                else:
                    alert_count = max(0, alert_count - 1)  # 서서히 감소

                alert = (alert_count >= args.persist)

                vis = frame_resized.copy()
                color = (0, 0, 255) if alert else (0, 255, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"[LOCKED] diff={diff:.3f} thr={args.diff_th:.2f} "
                                 f"persist {alert_count}/{args.persist}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if alert:
                    cv2.putText(vis, "ALERT: pattern changed!", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # --- 공통: 표시 & 스케줄러 ---
        
        cv2.imshow("openeye_cctv", vis)
        #cv2.imshow("camera_view", frame_resized)   # 원본 프레임 (ROI 박스 포함)
        #cv2.imshow("edge_debug", frm_edges)        # Canny 엣지 결과 (디버깅용)
        now = time.perf_counter()
        sleep_s = next_tick - now

        if sleep_s > 0:
            key = cv2.waitKey(int(sleep_s * 1000)) & 0xFF
        else:
            if DROP_LATE_FRAMES:
                late = -sleep_s
                while late > frame_period and cap.grab():
                    late -= frame_period
                    next_tick += frame_period
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

        next_tick += frame_period
        if key == 27:  # ESC
            break
        # R 키: 다시 SEARCH로 되돌아가 재잠금
        if key in (ord('r'), ord('R')):
            mode = "SEARCH"
            lock_xy = None
            alert_count = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()