# -*- coding: utf-8 -*-
# chair_monitor_local_template.py
# - 기준 템플릿과 다르면 '이상', 허용 박스 밖이면 '이탈'
# - 카메라 흔들림 억제: 로컬 검색 ROI + EMA + 연속 프레임 히스테리시스 + (옵션) ECC 안정화

import cv2, numpy as np, time, math
from typing import Tuple

# ===== 사용자 설정 =====
SOURCE = "../../assets/chair_test_480p_30fps.mp4"   # 0 이면 웹캠, 아니면 영상 경로
TEMPLATE_PATH = "../../assets/chair_template.png"   # 기준 템플릿(의자 정상 상태, 작게 잘라서 저장)

FRAME_MAX_W      = 720     # 입력 프레임을 이 너비로 축소(0이면 원본)
ANALYZE_EVERY    = 1        # N프레임마다 분석(1=매프레임, 2=절반 등)

# --- 위치 판정(허용 박스) ---
ALLOWED_BOX_SCALE = 0.3     # 허용 박스 크기(템플릿 폭/높이의 배수). 0.6~0.9 권장

# --- 로컬 검색 ROI ---
SEARCH_WIN_SCALE  = 1.2     # 검색창 크기(템플릿 폭/높이의 배수). 작을수록 튀지 않음
MIN_TRUST_SCORE   = 0.45    # 이 점수 미만이면 탐지 결과를 신뢰하지 않고 이전 중심 유지

# --- 모양(appearance) 판정 ---
PATCH_SCALE          = 1.6  # home/현재 패치 크기(템플릿의 배수)
APPEAR_SCORE_THRESH  = 0.55 # 모양 유사도 임계(T M_CCOEFF_NORMED). 낮출수록 덜 민감
APPEAR_FRAMES_REQ    = 6    # 모양 이탈 연속 프레임 요구(히스테리시스)

# --- 공통 히스테리시스/스무딩 ---
DEVIATE_FRAMES_REQ = 8      # (위치/모양) 이탈 연속 프레임 요구
EMA_ALPHA          = 0.20   # 중심점 지수이동평균(0.1~0.3)
GRACE_SECS         = 10     # 이탈 상태가 이 시간 이상 지속되면 ALERT

# --- 안정화 옵션 ---
STAB_SCALE = 0.5            # ECC 안정화(0=끄기, 0.4~0.7 권장)
METHOD     = cv2.TM_CCOEFF_NORMED
ROT_DEGS   = list(range(-15, 16, 5))  # 템플릿 회전 내성. 간격 줄이면 정확↑(속도↓)

# --- 오클루전(사람 통과) 게이트 ---
MOTION_GATE = True
MOTION_AREA_THRESH = 0.15   # 허용 박스 영역 대비 움직임 비율 임계 (0.10~0.25 권장)
MORPH_K = 3                 # 모폴로지 커널(노이즈 제거)

# ===================== 유틸리티 =====================

def resize_keep_w(img, max_w: int):
    if max_w and img.shape[1] > max_w:
        r = max_w / img.shape[1]
        return cv2.resize(img, (max_w, int(img.shape[0]*r))), r
    return img, 1.0

def rotate_img(img, deg):
    (h,w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_for_matching(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return g

def clamp_box(xc, yc, half_w, half_h, W, H):
    x1 = max(0, min(W-1, int(xc - half_w)))
    y1 = max(0, min(H-1, int(yc - half_h)))
    x2 = max(0, min(W,   int(xc + half_w)))
    y2 = max(0, min(H,   int(yc + half_h)))
    if x2 <= x1: x2 = min(W, x1 + 2)
    if y2 <= y1: y2 = min(H, y1 + 2)
    return x1, y1, x2, y2

def crop_patch(img, center, size_wh):
    (W,H) = (img.shape[1], img.shape[0])
    half_w, half_h = size_wh[0]//2, size_wh[1]//2
    x1,y1,x2,y2 = clamp_box(center[0], center[1], half_w, half_h, W, H)
    return img[y1:y2, x1:x2].copy(), (x1,y1,x2,y2)

def center_from(top_left, size):
    x,y = top_left
    tw, th = size
    return (x + tw//2, y + th//2)

def best_match_global(frame_gray, templates_gray):
    best = {"score": -1, "top_left": (0,0), "size": None, "deg": 0}
    for deg, tmpl in templates_gray:
        if frame_gray.shape[0] < tmpl.shape[0] or frame_gray.shape[1] < tmpl.shape[1]:
            continue
        res = cv2.matchTemplate(frame_gray, tmpl, METHOD)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv > best["score"]:
            th, tw = tmpl.shape[:2]
            best.update({"score": maxv, "top_left": maxloc, "size": (tw, th), "deg": deg})
    return best

def best_match_local(frame_gray, templates_gray, center, size_wh):
    W, H = frame_gray.shape[1], frame_gray.shape[0]
    half_w, half_h = size_wh[0]//2, size_wh[1]//2
    x1,y1,x2,y2 = clamp_box(center[0], center[1], half_w, half_h, W, H)
    roi = frame_gray[y1:y2, x1:x2]
    best = {"score": -1, "top_left": (0,0), "size": None, "deg": 0}
    for deg, tmpl in templates_gray:
        if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
            continue
        res = cv2.matchTemplate(roi, tmpl, METHOD)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv > best["score"]:
            th, tw = tmpl.shape[:2]
            best["score"] = maxv
            best["top_left"] = (maxloc[0] + x1, maxloc[1] + y1)
            best["size"] = (tw, th)
            best["deg"] = deg
    return best, (x1,y1,x2,y2)

def stabilize_ecc(prev_small, curr_small):
    """prev_small, curr_small: 다운스케일된 grayscale"""
    warp = np.eye(2, 3, dtype=np.float32)  # translation 모델
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
    try:
        _, warp = cv2.findTransformECC(prev_small, curr_small, warp,
                                       motionType=cv2.MOTION_TRANSLATION,
                                       criteria=criteria, inputMask=None, gaussFiltSize=3)
    except cv2.error:
        pass
    return warp

# ===================== 메인 =====================

def main():
    # 1) 템플릿 로드 & 회전군 준비
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        raise SystemExit(f"Cannot read template: {TEMPLATE_PATH}")
    template, _ = resize_keep_w(template, 300)  # 과대/과소 방지
    templates_gray = [(deg, preprocess_for_matching(rotate_img(template, deg)))
                      for deg in ROT_DEGS]

    # 2) 입력 소스
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {SOURCE}")

    # 3) 첫 프레임에서 홈(home) 초기화 (전역 매칭)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("No first frame from source")

    first, _ = resize_keep_w(first, FRAME_MAX_W)
    first_g = preprocess_for_matching(first)

    # 템플릿이 프레임보다 크면 축소
    th_tmp, tw_tmp = templates_gray[0][1].shape[:2]
    H0, W0 = first_g.shape[:2]
    if th_tmp > H0 or tw_tmp > W0:
        ratio = min(H0 / max(2, th_tmp), W0 / max(2, tw_tmp), 1.0)
        new_w = max(2, int(tw_tmp * ratio))
        new_h = max(2, int(th_tmp * ratio))
        templates_gray = [(deg, cv2.resize(tg, (new_w, new_h))) for deg, tg in templates_gray]

    bm0 = best_match_global(first_g, templates_gray)
    if bm0["score"] < 0.35:
        raise SystemExit(f"Init match too weak: score={bm0['score']:.2f}. "
                         f"Check TEMPLATE_PATH or adjust parameters.")

    home = center_from(bm0["top_left"], bm0["size"])
    tw0, th0 = bm0["size"]
    print(f"[INIT] score={bm0['score']:.3f}, home={home}, template=({tw0}x{th0}), frame=({W0}x{H0})")

    # 허용 박스(allowed ROI) 크기
    allowed_half_w = max(4, int(ALLOWED_BOX_SCALE * tw0))
    allowed_half_h = max(4, int(ALLOWED_BOX_SCALE * th0))

    # 모양 비교용 home_patch
    home_patch, home_box = crop_patch(
        first_g, home, (int(PATCH_SCALE*tw0), int(PATCH_SCALE*th0))
    )
    if home_patch.size == 0:
        home_patch, home_box = crop_patch(first_g, home, (tw0, th0))
        if home_patch.size == 0:
            raise SystemExit("home_patch crop failed — adjust PATCH_SCALE or recapture template.")

    # 상태 변수
    deviate_since = None
    deviate_run   = 0
    appear_run    = 0
    ema_center    = home
    prev_small    = None

    # 배경 차분기 (사람/큰 움직임 검출용)
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
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

        # (옵션) ECC 안정화(translation 보정)
        if STAB_SCALE and STAB_SCALE > 0:
            small = cv2.resize(fg, None, fx=STAB_SCALE, fy=STAB_SCALE)
            if prev_small is not None:
                warp = stabilize_ecc(prev_small, small)
                warp_big = warp.copy()
                warp_big[:,2] /= STAB_SCALE
                fg = cv2.warpAffine(fg, warp_big, (fg.shape[1], fg.shape[0]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            prev_small = small

        # 4) 로컬 검색 ROI에서 템플릿 매칭
        search_w = int(SEARCH_WIN_SCALE * tw0)
        search_h = int(SEARCH_WIN_SCALE * th0)
        center_for_search = ema_center if ema_center is not None else home
        bm, search_box = best_match_local(fg, templates_gray, center_for_search, (search_w, search_h))
        c_raw = center_from(bm["top_left"], bm["size"])
        score = bm["score"]

      
        # 신뢰 낮거나 오클루전이면 '중심 업데이트 보류'
        hold = (score < MIN_TRUST_SCORE) or occluded
        use_center = c_raw
        if hold:
            use_center = ema_center if ema_center is not None else home

        # EMA 스무딩
        ema_center = (
            int(EMA_ALPHA * use_center[0] + (1-EMA_ALPHA) * ema_center[0]),
            int(EMA_ALPHA * use_center[1] + (1-EMA_ALPHA) * ema_center[1])
        )
        # --- 오클루전 게이트: 허용 박스 안 '큰 움직임' 비율 계산 ---
        occluded = False
        if MOTION_GATE:
            # 원본 프레임(그레이 전)으로 마스크 만들면 그림자 영향이 덜함 → 여기선 fg(그레이)로도 충분
            m = bg.apply(frame)  # frame(컬러)나 fg(그레이) 중 선택, frame 사용 추천
            _, m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 허용 박스 영역만 잘라서 움직임 비율 측정
            W, H = fg.shape[1], fg.shape[0]
            x1a, y1a, x2a, y2a = clamp_box(home[0], home[1], allowed_half_w, allowed_half_h, W, H)
            m_crop = m[y1a:y2a, x1a:x2a]
            move_ratio = float(m_crop.sum()) / (255.0 * max(1, m_crop.size))

            occluded = (move_ratio >= MOTION_AREA_THRESH)
        # 5) 위치 판정: 허용 박스 밖이면 위치 이탈
        W, H = fg.shape[1], fg.shape[0]
        x1a, y1a, x2a, y2a = clamp_box(home[0], home[1], allowed_half_w, allowed_half_h, W, H)
        if occluded:
        # 사람/큰 움직임으로 가려진 프레임은 '위치 이탈' 판정 일시 중지
            pos_deviated = False
        else:
            pos_deviated = not (x1a <= ema_center[0] <= x2a and y1a <= ema_center[1] <= y2a)

        # 6) 모양 판정: home_patch vs 현재 패치
        curr_patch, curr_box = crop_patch(fg, ema_center, (int(PATCH_SCALE*tw0), int(PATCH_SCALE*th0)))
        app_score = 1.0
        app_deviated = False
        if curr_patch.size and home_patch.size and \
           (curr_patch.shape[0] >= 8 and curr_patch.shape[1] >= 8) and \
           (home_patch.shape[0] >= 8 and home_patch.shape[1] >= 8):

            hp = home_patch
            cp = curr_patch
            # 크기차이가 크면 home_patch를 현재 패치에 맞춰 축소
            if cp.shape[1] < hp.shape[1] or cp.shape[0] < hp.shape[0]:
                scale = min(cp.shape[1]/hp.shape[1], cp.shape[0]/hp.shape[0])
                if scale < 1.0:
                    new_w = max(2, int(hp.shape[1]*scale))
                    new_h = max(2, int(hp.shape[0]*scale))
                    hp = cv2.resize(hp, (new_w, new_h))

            if cp.shape[0] >= hp.shape[0] and cp.shape[1] >= hp.shape[1]:
                res = cv2.matchTemplate(cp, hp, METHOD)
                _, app_score, _, _ = cv2.minMaxLoc(res)
                app_deviated = (app_score < APPEAR_SCORE_THRESH)

        # 7) 연속 프레임 히스테리시스(위치 또는 모양 이탈)
        deviated_now = pos_deviated or app_deviated
        deviate_run = deviate_run + 1 if deviated_now else 0

        now = time.time()
        if deviate_run >= DEVIATE_FRAMES_REQ:
            if deviate_since is None:
                deviate_since = now
        else:
            deviate_since = None

        # ALERT 판정
        alert = False; remain = 0
        if deviate_since is not None:
            elapsed = now - deviate_since
            remain = max(0, int(GRACE_SECS - elapsed))
            if elapsed >= GRACE_SECS:
                alert = True

        # 8) 시각화
        vis = frame.copy()
        # 템플릿 매칭 사각형
        tw2, th2 = bm["size"] if bm["size"] else (0,0)
        x,y = bm["top_left"]
        if tw2 > 0 and th2 > 0:
            cv2.rectangle(vis, (x,y), (x+tw2, y+th2), (0,255,0), 2)

        # 허용 박스(노랑)
        cv2.rectangle(vis, (x1a,y1a), (x2a,y2a), (0,255,255), 2)
        # 검색창(보라)
        sx1,sy1,sx2,sy2 = search_box
        cv2.rectangle(vis, (sx1,sy1), (sx2,sy2), (160,100,255), 1)
        # 패치 박스(청록)
        px1,py1,px2,py2 = curr_box
        cv2.rectangle(vis, (px1,py1), (px2,py2), (50,200,200), 1)

        # 중심점/홈 표시
        cv2.circle(vis, home, 5, (255,255,0), -1)
        cv2.circle(vis, ema_center, 5, (0,0,255) if deviated_now else (0,255,0), -1)
        cv2.line(vis, home, ema_center, (0,0,255) if deviated_now else (0,255,0), 2)

        cv2.putText(vis, f"score={score:.2f} app={app_score:.2f}",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(vis, f"pos_dev={pos_deviated} app_dev={app_deviated} N={deviate_run}",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2)

        if deviate_since and not alert:
            cv2.putText(vis, f"Return within {remain}s",
                        (20,95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        if alert:
            cv2.putText(vis, "ALERT: Pattern deviated!",
                        (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        cv2.imshow("chair_monitor_local_template", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()