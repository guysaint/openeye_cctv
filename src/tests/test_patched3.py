#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB/소물체 감시 — 템플릿에서 물체 자동 분리(배경 제거) + 전역(gray)/로컬(edge) 매칭
창:
 - openeye_cctv : 최종 오버레이
 - camera_view  : 원본 프레임
 - edge_debug   : 엣지(백그라운드)

키:
 - R : 전역 재탐색(SEARCH)
 - ESC : 종료
"""
import cv2, numpy as np, argparse, time
from typing import Tuple

# ---------- utils ----------
def parse_source(v):
    # "0","1" 같은 문자열은 카메라 인덱스로, 그 외는 파일 경로로
    try:
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return v
    except:
        return v

def resize_keep_w(img, max_w:int)->Tuple:
    h, w = img.shape[:2]
    if max_w and w>max_w:
        s = max_w/float(w)
        return cv2.resize(img, (int(w*s), int(h*s))), s
    return img, 1.0

def edges_clean(
    bgr,
    blur=None,
    k1=None, k2=None,
    op: str = "none",
    ksize: int = 3,
    open_iter: int = 1,
    close_iter: int = 0,
    min_area: int = 0,
    mode: str = "soft",
    kill_hline: bool = False,
):
    """모드별 안전한 엣지 추출 + 실패시 폴백."""
    # ---- 모드 프리셋 ----
    if mode == "raw":
        blur = 3 if blur is None else blur
        k1 = 0.10 if k1 is None else k1
        k2 = 0.10 if k2 is None else k2
        op = "none"
        kill_hline = False
    elif mode == "soft":
        blur = 5 if blur is None else blur
        k1 = 0.20 if k1 is None else k1
        k2 = 0.20 if k2 is None else k2
        op = "opening" if op == "none" else op
        kill_hline = True if kill_hline is False else kill_hline
    else:  # "strong"
        blur = 5 if blur is None else blur
        k1 = 0.30 if k1 is None else k1
        k2 = 0.30 if k2 is None else k2
        op = "opening"
        kill_hline = True if kill_hline is False else kill_hline

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur: g = cv2.GaussianBlur(g, (blur, blur), 0)

    med = float(np.median(g))
    lo = int(max(0, (1.0 - k1) * med))
    hi = int(min(255, (1.0 + k2) * med))
    e = cv2.Canny(g, lo, hi)

    # ---- 엣지가 너무 적으면(진짜 까만 창) 안전 폴백 ----
    nz = int((e > 0).sum())
    if nz < 100:   # 화면 크기와 무관하게 아주 작은 기준
        e = cv2.Canny(g, 50, 150)     # 고정 임계 폴백
        if op == "none":
            op = "opening"            # 살짝만 정리

    if op != "none":
        K = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        if "opening" in op:
            e = cv2.morphologyEx(e, cv2.MORPH_OPEN, K, iterations=open_iter)
        if "closing" in op:
            e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, K, iterations=close_iter)
            if ksize >= 3 and close_iter > 0:
                e = cv2.erode(e, K, iterations=1)

    if kill_hline:
        e = cv2.morphologyEx(
            e, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
            iterations=1
        )

    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (e > 0).astype(np.uint8), connectivity=8)
        mask = np.zeros_like(e)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255
        e = mask
    return e

def rotate_edge(img, ang):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), ang, 1.0)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_NEAREST, borderValue=0)

# ---------- template processing ----------
def make_template_from_image(path):
    """
    1) 템플릿 이미지를 읽고, 2) Otsu + 모폴로지로 전경(USB)을 분리,
    3) 가장 큰 컨투어로 tight crop, 4) 엣지 + 마스크 생성
    """
    bgr = cv2.imread(path)
    if bgr is None: raise SystemExit(f"[Error] Cannot read template: {path}")
    # 너무 크면 축소
    bgr, _ = resize_keep_w(bgr, 640)

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 밝은 책상 + 어두운 USB 가정 → INV+OTSU
    _, fg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    K = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, K, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, K, iterations=1)

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # 실패시 대비: 중앙 80% 박스 사용
        h,w = g.shape[:2]; x0,y0 = w//10, h//10; x1,y1 = w-w//10, h-h//10
        mask = np.zeros_like(g); mask[y0:y1, x0:x1] = 255
    else:
        cmax = max(cnts, key=cv2.contourArea)
        mask = np.zeros_like(g)
        cv2.drawContours(mask, [cmax], -1, 255, -1)
    # tight crop
    ys, xs = np.where(mask>0)
    y0,y1 = ys.min(), ys.max()+1; x0,x1 = xs.min(), xs.max()+1
    bgr = bgr[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 엣지는 마스크로 제한
    edges = edges_clean(bgr, op="closing", open_iter=1, close_iter=1, blur=3, k1=0.4, k2=0.4)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    return bgr, gray, edges, (mask>0).astype(np.uint8)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default="0", help="카메라 인덱스('0','1'...) 또는 영상경로")
    p.add_argument("--template", type=str, required=True, help="템플릿 이미지(USB 잘라낸/혹은 원본)")
    p.add_argument("--maxw", type=int, default=1280)
    p.add_argument("--sim-th", type=float, default=0.55)
    p.add_argument("--move-tol", type=int, default=12)
    p.add_argument("--diff-th", type=float, default=0.12)
    p.add_argument("--persist", type=int, default=6)
    p.add_argument("--search-radius", type=int, default=90)
    p.add_argument("--edge-mode", type=str, default="soft",
               choices=["raw","soft","strong"],
               help="Canny/모폴로지 강도 (raw=가장 약함)")
    p.add_argument("--edge-k1", type=float, default=None,
                help="Canny lo 임계 비율(미디언 기반). None이면 모드에 따름")
    p.add_argument("--edge-k2", type=float, default=None,
                help="Canny hi 임계 비율(미디언 기반). None이면 모드에 따름")
    return p.parse_args()

# ---------- main ----------
def main():
    args = parse_args()

    # source open (camera or file)
    cap = cv2.VideoCapture(parse_source(args.video))
    if not cap.isOpened():
        raise SystemExit(f"[Error] Cannot open source: {args.video}")

    # template build
    t_bgr, t_gray, t_edges, t_mask = make_template_from_image(args.template)
    t_edges = edges_clean(
    t_bgr,
    op="closing",
    open_iter=1,
    close_iter=1,
    blur=3,
    k1=0.2,
    k2=0.2,
    mode="soft",
    kill_hline=False
)
    th, tw = t_edges.shape[:2]
    print(f"[Info] Template(after crop): {tw}x{th}")

    # variants (rot/scale)
    angles = [-6,-3,0,3,6]
    scales = [0.95, 1.0, 1.05]
    variants = []
    for s in scales:
        te0 = cv2.resize(t_edges, (max(2,int(tw*s)), max(2,int(th*s))), interpolation=cv2.INTER_NEAREST)
        tm0 = cv2.resize(t_mask,  (te0.shape[1], te0.shape[0]), interpolation=cv2.INTER_NEAREST)
        for ang in angles:
            te = rotate_edge(te0, ang)
            tm = rotate_edge(tm0, ang)
            hh, ww = te.shape[:2]
            variants.append((s, ang, te, (tm>0).astype(np.uint8), ww, hh))

    # fps scheduler
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    period = 1.0/fps; next_tick = time.perf_counter()+period

    mode = "SEARCH"
    lock_xy = None
    cx_lock = cy_lock = None
    alert_count = 0
    ema_sim = ema_diff = None
    EMA_A = 0.25
    alert_state = False
    cooldown = 0
    MIN_HOLD = 10
    R_MIN = int(args.search_radius); R_MAX = 240; R = R_MIN
    no_local = 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        show, _ = resize_keep_w(frame, args.maxw)
        H, W = show.shape[:2]
        edges = edges_clean(
        show,
        mode=args.edge_mode   # 인자로 모드 전달
        )

        if th>H or tw>W:
            vis = show.copy()
            cv2.putText(vis, "Template larger than frame", (20,40), 0, 0.8, (0,0,255), 2)
        else:
            if mode=="SEARCH":
                gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED, mask=t_mask)
                res = np.nan_to_num(res, nan=-1.0)
                _, mx, _, ml = cv2.minMaxLoc(res)
                x1,y1 = int(ml[0]), int(ml[1]); x2,y2 = x1+tw, y1+th
                vis = show.copy()
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(vis, f"[SEARCH] score={mx:.3f}", (20,30), 0, 0.8, (0,255,0), 2)
                if mx>=0.5:
                    mode="LOCKED"
                    lock_xy=(x1,y1)
                    cx_lock, cy_lock = x1+tw//2, y1+th//2
                    alert_count=0; ema_sim=ema_diff=None; alert_state=False; cooldown=0; R=R_MIN; no_local=0
            else:
                # diff on locked box
                x1l,y1l = lock_xy
                x1l = max(0,min(W-tw,x1l)); y1l=max(0,min(H-th,y1l))
                roi_l = edges[y1l:y1l+th, x1l:x1l+tw]
                diff_l = float(cv2.mean(cv2.absdiff(roi_l, t_edges))[0])/255.0
                if ema_diff is None: ema_diff = diff_l
                ema_diff = (1-EMA_A)*ema_diff + EMA_A*diff_l

                # local search window
                x0 = max(0, cx_lock - R); y0 = max(0, cy_lock - R)
                xE = min(W, cx_lock + R); yE = min(H, cy_lock + R)
                search = edges[y0:yE, x0:xE]
                sH,sW = search.shape[:2]

                # rotate/scale variants
                best=None
                for s,ang,te,tm,ww,hh in variants:
                    if hh>sH or ww>sW: continue
                    r = cv2.matchTemplate(search, te, cv2.TM_CCORR_NORMED, mask=tm)
                    if r.size==0: continue
                    r = np.nan_to_num(r, nan=-1.0)
                    _, mx, _, ml = cv2.minMaxLoc(r)
                    sx,sy = int(ml[0]), int(ml[1])
                    x1b,y1b = x0+sx, y0+sy
                    x2b,y2b = x1b+ww, y1b+hh
                    cand = (float(mx), x1b,y1b,x2b,y2b, ang, s, sx, sy)
                    if best is None or cand[0]>best[0]: best=cand

                vis = show.copy()
                if best is None:
                    no_local += 1
                    R = min(R_MAX, int(R*1.35))
                    bad = (ema_diff >= args.diff_th)
                    box = (x1l,y1l,x1l+tw,y1l+th)
                    info = f"[LOCKED] no local match | diff(ema)={ema_diff:.3f}"
                else:
                    no_local = 0
                    scoreN,x1b,y1b,x2b,y2b,ang,sc,sx,sy = best
                    cx_b, cy_b = (x1b+x2b)//2, (y1b+y2b)//2
                    move_px = int(((cx_b-cx_lock)**2 + (cy_b-cy_lock)**2)**0.5)
                    # EMA(sim)
                    if ema_sim is None: ema_sim = scoreN
                    ema_sim = (1-EMA_A)*ema_sim + EMA_A*scoreN
                    hit_border = (sx<=1 or sy<=1 or sx>=sW-1-(x2b-x1b) or sy>=sH-1-(y2b-y1b))
                    bad = (ema_sim<args.sim_th) or (move_px>args.move_tol) or (ema_diff>=args.diff_th) or hit_border
                    if not bad:
                        R = max(R_MIN, int(R*0.92))
                        # follow smoothly
                        ALPHA=0.3
                        cx_lock = int((1-ALPHA)*cx_lock + ALPHA*cx_b)
                        cy_lock = int((1-ALPHA)*cy_lock + ALPHA*cy_b)
                        lock_xy = (max(0,min(W-tw, cx_lock - tw//2)),
                                   max(0,min(H-th, cy_lock - th//2)))
                    box = (x1b,y1b,x2b,y2b)
                    info = (f"[LOCKED] sim(ema)={ema_sim:.3f} thr={args.sim_th:.2f} | "
                            f"move={move_px}px tol={args.move_tol} | "
                            f"diff(ema)={ema_diff:.3f} | ang={ang} sc={sc:.2f}")

                if bad: alert_count += 1
                else:   alert_count = max(0, alert_count-1)
                want = (alert_count >= args.persist)
                if cooldown==0 and want!=alert_state:
                    alert_state = want; cooldown = MIN_HOLD
                if cooldown>0: cooldown -= 1

                color = (0,0,255) if alert_state else ((0,200,255) if best is None else (0,255,0))
                x1d,y1d,x2d,y2d = box
                cv2.rectangle(vis,(x1d,y1d),(x2d,y2d),color,2)
                cv2.putText(vis, f"{info} | persist {alert_count}/{args.persist}", (20,30), 0, 0.7, color, 2)
                cv2.rectangle(vis,(x0,y0),(xE,yE),(0,150,0),1)

                if best is None and no_local>=12:
                    mode="SEARCH"; R=R_MIN
                    ema_sim=ema_diff=None; alert_state=False; cooldown=0; alert_count=0

        # show windows
        cv2.imshow("openeye_cctv", vis)
        cv2.imshow("camera_view", show)
        cv2.imshow("edge_debug", edges)

        # scheduler
        now = time.perf_counter(); sleep = next_tick - now
        key = cv2.waitKey(max(1,int(max(0,sleep)*1000))) & 0xFF
        next_tick += period
        if key==27: break
        if key in (ord('r'),ord('R')):
            mode="SEARCH"; alert_count=0; ema_sim=ema_diff=None
            alert_state=False; cooldown=0; no_local=0; R=R_MIN

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()