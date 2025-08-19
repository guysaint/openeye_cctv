#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2b — 엣지 기반 템플릿 매칭 + FPS 스케줄러 (LOCK 후 변화 감지, 초록 안정 표시 튜닝)
- SEARCH: 전체 프레임에서 템플릿 위치 찾기
- LOCKED: 로컬 검색(회전/스케일 보정) + 이동/유사도/패턴차이 기반 판정
- Anti-flicker: EMA + 래치(히스테리시스) + 쿨다운 + persist + 워밍업
- 디버그 창:
    openeye_cctv : 최종 오버레이(상태/점수)
    camera_view  : 원본 프레임
    edge_debug   : 엣지(백그라운드 처리)
"""

import cv2
import numpy as np
import argparse
from typing import Tuple
import time

# ---------------- 유틸 ----------------
def resize_keep_w(img, max_w: int) -> Tuple:
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
    return table.get(name.upper(), cv2.TM_CCORR)

def edges_clean(
    bgr,
    blur=3,
    k1=0.66,
    k2=1.33,
    op: str = "none",   # "opening" | "closing" | "both" | "none"
    ksize: int = 3,
    open_iter: int = 1,
    close_iter: int = 1,
    min_area: int = 0
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
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255
        e = mask

    return e

def rotate_edge(img, angle_deg: float):
    """엣지 맵(단일 채널)을 angle_deg 만큼 회전"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

# ---------------- 메인 ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Edge-based template matching with FPS scheduler (lock & change detection)")
    # p.add_argument("--video", type=str, default="../../assets/chair_test_480p_30fps.mp4",
    #                help="영상 경로 또는 0(웹캠). 기본: ../../assets/chair_test_480p_30fps.mp4")
    # p.add_argument("--template", type=str, default="../../assets/chair_template.png",
    #                help="템플릿 이미지 경로 (의자 ROI)")
    p.add_argument("--video", type=str, default=1,
                   help="영상 경로 또는 0(웹캠). 기본: ../../assets/chair_test_480p_30fps.mp4")
    p.add_argument("--template", type=str, default="../../assets/usb_test2.png",
                   help="템플릿 이미지 경로 (의자 ROI)")
    p.add_argument("--maxw", type=int, default=1280,
                   help="프레임 표시/처리용 최대 가로폭 (속도/가독성). 기본: 1280")
    p.add_argument("--method", type=str, default="TM_CCORR",
                   help="SEARCH 단계 매칭 방법(기본 TM_CCORR)")
    # 형태학 파라미터
    p.add_argument("--tmpl-op", type=str, default="closing",
                   help="템플릿: none|opening|closing|both (기본: closing)")
    p.add_argument("--tmpl-min-area", type=int, default=10,
                   help="템플릿: 작은 조각 제거 임계(픽셀). 기본: 10")
    p.add_argument("--frm-op", type=str, default="none",
                   help="프레임: none|opening|closing|both (기본: none)")
    p.add_argument("--frm-min-area", type=int, default=0,
                   help="프레임: 작은 조각 제거 임계(픽셀). 기본: 0")

    # 변화/유사도 판정 파라미터 (완화 기본값)
    p.add_argument("--diff-th", type=float, default=0.20,
                   help="변화 감지 임계 (0~1). 값↑=더 엄격")
    p.add_argument("--persist", type=int, default=6,
                   help="이상 후보가 연속 N프레임 지속될 때 경고")
    p.add_argument("--sim-th", type=float, default=0.50,
                   help="LOCKED 유사도 임계(0..1). ↓이면 이상 후보")
    p.add_argument("--move-tol", type=int, default=20,
                   help="LOCKED 허용 이동량(px). 초과면 이상 후보")
    p.add_argument("--search-radius", type=int, default=110,
                   help="LOCKED 로컬 탐색 반경(px)")
    return p.parse_args()

def main():
    args = parse_args()

    # 입력/템플릿 로드
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
    th, tw = tmpl_edges.shape[:2]
    tmpl_mask = (tmpl_edges > 0).astype(np.uint8)
    print(f"[Info] Template edge size: {tw}x{th}")

    # --- 회전 + 스케일 템플릿 프리컴퓨트 ---
    angles = [-8, -4, 0, 4, 8]
    scales = [0.92, 0.96, 1.0, 1.04, 1.08]
    tmpl_variants = []  # (scale, angle, edge, mask, w, h)
    for s in scales:
        te0 = cv2.resize(tmpl_edges, (max(2, int(tw*s)), max(2, int(th*s))), interpolation=cv2.INTER_NEAREST)
        for ang in angles:
            te = rotate_edge(te0, ang)
            tm = (te > 0).astype(np.uint8)
            hh, ww = te.shape[:2]
            tmpl_variants.append((s, ang, te, tm, ww, hh))

    method_search = get_method(args.method)
    print(f"[Info] SEARCH method: {args.method}")

    # FPS 스케줄러
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    frame_period = 1.0 / fps
    next_tick = time.perf_counter() + frame_period
    DROP_LATE_FRAMES = True
    delay_ms = int(frame_period * 1000)
    print(f"[Info] Source FPS: {fps:.2f} (period={delay_ms} ms)")

    # ----- 상태 / Anti-flicker 변수 -----
    mode = "SEARCH"
    lock_xy = None
    cx_lock = cy_lock = None
    alert_count = 0
    ema_sim = None
    ema_diff = None
    EMA_A = 0.25
    alert_state = False
    state_cooldown = 0
    MIN_HOLD = 12
    no_local_frames = 0
    frame_idx = 0
    warmup = 0                 # 잠금 직후 워밍업 프레임

    # 로컬 탐색 반경 적응
    R_MIN = int(args.search_radius)
    R_MAX = 240
    R_cur = R_MIN
    FAIL_EXPAND = 1.35
    SUCC_SHRINK = 0.92

    # 주기적 전역 재정렬(앵커링)
    RESEEK_EVERY = 30

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
                # --- 전역 매칭 ---
                res = cv2.matchTemplate(frm_edges, tmpl_edges, method_search, mask=tmpl_mask)
                res = np.nan_to_num(res, nan=-1.0, posinf=1e9, neginf=-1e9)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if method_search in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    top_left = min_loc; score = 1.0 - float(min_val)
                else:
                    top_left = max_loc; score = float(max_val)

                x1, y1 = top_left
                x2, y2 = x1 + tw, y1 + th

                if score >= 0.35:  # ← 완화
                    lock_xy = (x1, y1)
                    cx_lock = x1 + tw // 2
                    cy_lock = y1 + th // 2
                    mode = "LOCKED"
                    # 초기화
                    ema_sim = None
                    ema_diff = None
                    alert_count = 0
                    alert_state = False
                    state_cooldown = 0
                    no_local_frames = 0
                    R_cur = R_MIN
                    warmup = 10  # ← 워밍업 10프레임: 알람 금지

                vis = frame_resized.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"[SEARCH] score={score:.3f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                # --- LOCKED ---
                frame_idx += 1

                # (A) 잠금 ROI에서의 엣지 차이(0..1)
                x1_lock, y1_lock = lock_xy
                x1_lock = max(0, min(W - tw, x1_lock)); y1_lock = max(0, min(H - th, y1_lock))
                x2_lock, y2_lock = x1_lock + tw, y1_lock + th
                roi_lock_edges = frm_edges[y1_lock:y2_lock, x1_lock:x2_lock]
                diff_lock = float(cv2.mean(cv2.absdiff(roi_lock_edges, tmpl_edges))[0]) / 255.0

                # EMA 업데이트(차이)
                if ema_diff is None: ema_diff = diff_lock
                ema_diff = (1 - EMA_A) * ema_diff + EMA_A * diff_lock

                # (B) 적응형 로컬 탐색창
                roi_x0 = max(0, cx_lock - R_cur)
                roi_y0 = max(0, cy_lock - R_cur)
                roi_x1 = min(W, cx_lock + R_cur)
                roi_y1 = min(H, cy_lock + R_cur)
                search = frm_edges[roi_y0:roi_y1, roi_x0:roi_x1]
                sH, sW = search.shape[:2]

                # (C) 회전/스케일 템플릿 중 최고 후보
                best = None  # (scoreN, x1,y1,x2,y2, ang, scale, sx, sy)
                if sH > 2 and sW > 2:
                    for s, ang, te, tm, ww, hh in tmpl_variants:
                        if hh > sH or ww > sW:
                            continue
                        res = cv2.matchTemplate(search, te, cv2.TM_CCORR_NORMED, mask=tm)
                        if res.size == 0:
                            continue
                        res = np.nan_to_num(res, nan=-1.0)
                        _, mx, _, ml = cv2.minMaxLoc(res)
                        sx, sy = int(ml[0]), int(ml[1])
                        x1b = roi_x0 + sx;  y1b = roi_y0 + sy
                        x2b = x1b + ww;     y2b = y1b + hh
                        cand = (float(mx), x1b, y1b, x2b, y2b, ang, s, sx, sy)
                        if (best is None) or (cand[0] > best[0]):
                            best = cand

                vis = frame_resized.copy()

                # (D) 주기적 전역 재정렬(앵커링)
                if frame_idx % RESEEK_EVERY == 0:
                    resG = cv2.matchTemplate(frm_edges, tmpl_edges, method_search, mask=tmpl_mask)
                    resG = np.nan_to_num(resG, nan=-1.0)
                    _, mxG, _, mlG = cv2.minMaxLoc(resG)
                    gx, gy = int(mlG[0]), int(mlG[1])
                    if mxG > 0.40:
                        gcx, gcy = gx + tw//2, gy + th//2
                        cx_lock = int(0.7*cx_lock + 0.3*gcx)
                        cy_lock = int(0.7*cy_lock + 0.3*gcy)
                        lock_xy = (max(0, min(W - tw, cx_lock - tw//2)),
                                   max(0, min(H - th, cy_lock - th//2)))

                # (E) 후보 평가 + 반경/상태 업데이트
                if best is None:
                    # 로컬 매칭 실패: 반경 키우기 및 diff 기반 표시
                    no_local_frames += 1
                    R_cur = min(R_MAX, int(R_cur * FAIL_EXPAND))
                    bad_candidate = (ema_diff >= args.diff_th) and (warmup == 0)

                    box = (x1_lock, y1_lock, x2_lock, y2_lock)
                    info = f"[LOCKED] no local match | diff(ema)={ema_diff:.3f}"
                else:
                    no_local_frames = 0
                    scoreN, x1b, y1b, x2b, y2b, ang, sc, sx, sy = best
                    cx_best = (x1b + x2b)//2
                    cy_best = (y1b + y2b)//2
                    move_px = int(((cx_best - cx_lock)**2 + (cy_best - cy_lock)**2)**0.5)

                    # EMA(유사도)
                    if ema_sim is None: ema_sim = scoreN
                    ema_sim = (1 - EMA_A) * ema_sim + EMA_A * scoreN

                    # 검색창 가장자리 히트면 반경만 확장(이상 판정 X)
                    ww = x2b - x1b; hh = y2b - y1b
                    hit_border = (sx <= 2 or sy <= 2 or sx >= sW - 2 - ww or sy >= sH - 2 - hh)
                    if hit_border:
                        R_cur = min(R_MAX, int(R_cur * FAIL_EXPAND))

                    # ‘나쁨’ 후보? (워밍업 중엔 항상 정상 취급)
                    bad_candidate = (warmup == 0) and (
                        (ema_sim < args.sim_th) or
                        (move_px > args.move_tol) or
                        (ema_diff >= args.diff_th)
                    )

                    # 성공이면 반경 서서히 축소
                    if not bad_candidate:
                        R_cur = max(R_MIN, int(R_cur * SUCC_SHRINK))

                    # 정상일 때만 lock 중심을 따라가며 드리프트 억제
                    if not bad_candidate and not alert_state:
                        ALPHA = 0.3
                        cx_lock = int((1-ALPHA)*cx_lock + ALPHA*cx_best)
                        cy_lock = int((1-ALPHA)*cy_lock + ALPHA*cy_best)
                        lock_xy = (max(0, min(W - tw, cx_lock - tw//2)),
                                   max(0, min(H - th, cy_lock - th//2)))

                    # 표시 정보
                    box = (x1b, y1b, x2b, y2b)
                    info = (f"[LOCKED] sim(ema)={ema_sim:.3f} thr={args.sim_th:.2f} | "
                            f"move={move_px}px tol={args.move_tol} | "
                            f"diff(ema)={ema_diff:.3f} | ang={ang} sc={sc:.2f}")

                # persist 카운트 & 래치(쿨다운)
                if bad_candidate:
                    alert_count += 1
                else:
                    alert_count = max(0, alert_count - 1)

                want_state = (alert_count >= args.persist)
                if state_cooldown == 0 and want_state != alert_state:
                    alert_state = want_state
                    state_cooldown = MIN_HOLD
                if state_cooldown > 0:
                    state_cooldown -= 1

                # 워밍업 카운트다운
                if warmup > 0:
                    warmup -= 1
                    alert_state = False
                    alert_count = 0  # 워밍업 동안 강제로 정상 유지

                # 색상 결정 및 표시
                color = (0,0,255) if alert_state else ((0,200,255) if best is None else (0,255,0))
                x1d, y1d, x2d, y2d = box
                cv2.rectangle(vis, (x1d, y1d), (x2d, y2d), color, 2)
                mode_str = f"{mode} (R={R_cur})"
                cv2.putText(vis, f"{mode_str} | {info} | persist {alert_count}/{args.persist}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # 탐색창 외곽선
                cv2.rectangle(vis, (roi_x0, roi_y0), (roi_x1, roi_y1), (0,150,0), 1)

                # 연속 실패가 길면 SEARCH로 복귀
                if best is None and no_local_frames >= 12:
                    mode = "SEARCH"
                    no_local_frames = 0
                    R_cur = R_MIN
                    ema_sim = ema_diff = None
                    alert_state = False
                    alert_count = 0
                    state_cooldown = 0

        # --- 표시 & 스케줄러 ---
        cv2.imshow("openeye_cctv", vis)          # 최종 오버레이
        cv2.imshow("camera_view", frame_resized)  # 일반 화면
        cv2.imshow("edge_debug", frm_edges)       # 엣지(백그라운드)

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
        if key in (ord('r'), ord('R')):  # 재탐색
            mode = "SEARCH"
            lock_xy = None
            cx_lock = cy_lock = None
            alert_count = 0
            ema_sim = ema_diff = None
            alert_state = False
            state_cooldown = 0
            no_local_frames = 0
            R_cur = R_MIN
            warmup = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()