#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2b — 엣지 기반 템플릿 매칭 + 원래 FPS로 재생 (정리 버전)
- 엣지(Canny) + 형태학 정리 후 템플릿 매칭
- FPS 스케줄러로 재생 속도 1x 유지 (늦으면 프레임 드롭 옵션)
- 전역 검색 버전 (로컬 검색은 다음 단계에서 추가)

기본 경로:
  영상  : ../img/chair_test_480p_30fps.mp4
  템플릿: /img/chair_template.png
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

# 엣지 + 형태학 정리
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
    lo = int(max(0, (1.0 - k1) * med))   # ← 인자 k1 사용
    hi = int(min(255, (1.0 + k2) * med)) # ← 인자 k2 사용
    e = cv2.Canny(g, lo, hi)

    if op != "none":
        K = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        if op in ("opening", "both"):
            e = cv2.morphologyEx(e, cv2.MORPH_OPEN,  K, iterations=open_iter)
        if op in ("closing", "both"):
            e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, K, iterations=close_iter)
            if ksize >= 3:
                e = cv2.erode(e, K, iterations=1)  # 너무 굵어지면 살짝 얇게

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
    p = argparse.ArgumentParser(description="Edge-based template matching with FPS scheduler")
    # p.add_argument("--video", type=str, default="../../assets/chair_test_480p_30fps.mp4",
    #                help="영상 경로 또는 0(웹캠). 기본: ....//assets/chair_test_480p_30fps.mp4")
    # p.add_argument("--template", type=str, default="../../assets/chair_template.png",
    #                help="템플릿 이미지 경로 (의자 ROI). 기본: chair_template.png")
    p.add_argument("--video", type=str, default=1,
                   help="영상 경로 또는 0(웹캠). 기본: ....//assets/chair_test_480p_30fps.mp4")
    p.add_argument("--template", type=str, default="../../assets/usb_test2.jpg",
                   help="템플릿 이미지 경로 (의자 ROI). 기본: chair_template.png")
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
                   help="프레임: none|opening|closing|both (기본: none — 네 세팅)")
    p.add_argument("--frm-min-area", type=int, default=0,
                   help="프레임: 작은 조각 제거 임계(픽셀). 기본: 0")
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

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_resized, _ = resize_keep_w(frame, args.maxw)

        # 프레임 엣지
        frm_edges = edges_clean(
            frame_resized, op=args.frm_op, ksize=3, open_iter=1, close_iter=1, min_area=args.frm_min_area
        )

        H, W = frm_edges.shape[:2]
        if th > H or tw > W:
            vis = cv2.cvtColor(frm_edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, "Template larger than frame — use smaller template.",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 아래 스케줄러 공통 로직으로 이동
        else:
            # 엣지 픽셀 충분한지 체크 → 부족하면 그레이스로 안전매칭
            MIN_EDGE_PIX = 50
            nonzero_frm = int((frm_edges > 0).sum())
            nonzero_tmpl = int((tmpl_edges > 0).sum())

            if nonzero_frm < MIN_EDGE_PIX or nonzero_tmpl < MIN_EDGE_PIX:
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
                # (필요 시) 템플릿 축소
                gh, gw = gray.shape[:2]
                gth, gtw = tmpl_gray.shape[:2]
                if gth > gh or gtw > gw:
                    s = min(gh / gth, gw / gtw, 1.0)
                    tmpl_gray = cv2.resize(tmpl_gray, (int(gtw * s), int(gth * s)))
                    gth, gtw = tmpl_gray.shape[:2]
                res = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCORR)
                _, score, _, top_left = cv2.minMaxLoc(res)
                x1, y1 = top_left
                x2, y2 = x1 + gtw, y1 + gth
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                # 엣지 + 마스크 매칭 (정규화 없는 CCORR 기본)
                res = cv2.matchTemplate(frm_edges, tmpl_edges, method, mask=tmpl_mask)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    score = 1.0 - float(min_val)
                    top_left = min_loc
                else:
                    score = float(max_val)
                    top_left = max_loc
                x1, y1 = top_left
                x2, y2 = x1 + tw, y1 + th
                vis = cv2.cvtColor(frm_edges, cv2.COLOR_GRAY2BGR)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"score={score:.3f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(vis, f"delay={delay_ms}ms", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)

        # --- 공통: 표시 & 스케줄러 ---
        cv2.imshow("edge_template_match", vis)
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
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()