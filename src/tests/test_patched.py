import cv2, numpy as np, time, math
from typing import Tuple

# ===== 사용자 설정 =====
SOURCE = "../../assets/chair_test_480p_30fps.mp4"  # 0 이면 웹캠
TEMPLATE_PATH = "../../assets/chair_template.png"  # 기준 템플릿 파일
FRAME_MAX_W = 720          # 입력 프레임 축소 폭(속도 ↑), 0이면 원본
ANALYZE_EVERY = 2          # N프레임마다 분석(1=매프레임)
SCORE_THRESH  = 0.52       # 유사도 임계 (0~1)
GRACE_SECS    = 10         # 이탈 지속 허용 시간
ROT_DEGS      = list(range(-15, 16, 5)) # 템플릿 회전 각도(도)
METHOD        = cv2.TM_CCOEFF_NORMED     # 매칭 방법

# ---- 흔들림 완화 파라미터 ----
POS_THRESH_PX_BASE = 30    # 최소 위치 임계(픽셀). 템플릿 대각선 15%와 비교해 더 큰값 사용
DEVIATE_FRAMES_REQ = 90     # 연속 N프레임 이상 이탈이어야 이탈 시작으로 인정
EMA_ALPHA = 0.2            # 중심점 지수이동평균(0.1~0.3 권장)
STAB_SCALE = 0.5           # ECC 안정화용 다운스케일 비율(0=끄기)

# =====================

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

def best_match(frame_gray, templates_gray):
    best = {"score": -1, "top_left": (0,0), "size": None, "deg": 0}
    for deg, tmpl in templates_gray:
        res = cv2.matchTemplate(frame_gray, tmpl, METHOD)
        minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
        if maxv > best["score"]:
            th, tw = tmpl.shape[:2]
            best.update({
                "score": maxv,
                "top_left": maxloc,
                "size": (tw, th),
                "deg": deg
            })
    return best

def center_from(top_left, size):
    x,y = top_left
    tw, th = size
    return (x + tw//2, y + th//2)

def stabilize_ecc(prev_small, curr_small):
    """prev_small, curr_small: 작은 크기의 그레이 이미지"""
    warp = np.eye(2, 3, dtype=np.float32)  # translation
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
    try:
        cc, warp = cv2.findTransformECC(prev_small, curr_small, warp,
                                        motionType=cv2.MOTION_TRANSLATION,
                                        criteria=criteria, inputMask=None, gaussFiltSize=3)
    except cv2.error:
        pass
    return warp

def main():
    # 템플릿 준비
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        raise SystemExit(f"Cannot read template: {TEMPLATE_PATH}")
    template, _ = resize_keep_w(template, 300)
    templates_gray = [(deg, preprocess_for_matching(rotate_img(template, deg)))
                      for deg in ROT_DEGS]

    # 템플릿 대각선 기반 데드밴드 설정
    th, tw = template.shape[:2]
    template_diag = (tw**2 + th**2) ** 0.5
    POS_THRESH_PX = max(POS_THRESH_PX_BASE, int(0.15 * template_diag))

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {SOURCE}")

    # 기준 위치 추정
    ok, first = cap.read()
    if not ok:
        raise SystemExit("No first frame")
    first, scale_r = resize_keep_w(first, FRAME_MAX_W)
    first_g = preprocess_for_matching(first)
    bm0 = best_match(first_g, templates_gray)
    home = center_from(bm0["top_left"], bm0["size"])
    home_deg = bm0["deg"]
    print(f"[INIT] score={bm0['score']:.3f}, home={home}, deg={home_deg}, POS_THRESH={POS_THRESH_PX}px")

    deviate_since = None
    deviate_run = 0
    ema_center = None
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

        # --- 전역 안정화(ECC, 선택) ---
        if STAB_SCALE and STAB_SCALE > 0:
            small = cv2.resize(fg, None, fx=STAB_SCALE, fy=STAB_SCALE)
            if prev_small is not None:
                warp = stabilize_ecc(prev_small, small)
                warp_big = warp.copy()
                warp_big[:,2] /= STAB_SCALE   # translation 성분 스케일업
                fg = cv2.warpAffine(fg, warp_big, (fg.shape[1], fg.shape[0]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            prev_small = small

        # --- 템플릿 매칭 ---
        bm = best_match(fg, templates_gray)
        c  = center_from(bm["top_left"], bm["size"])
        score = bm["score"]

        # --- EMA로 중심 스무딩 ---
        if ema_center is None:
            ema_center = c
        else:
            ema_center = (
                int(EMA_ALPHA*c[0] + (1-EMA_ALPHA)*ema_center[0]),
                int(EMA_ALPHA*c[1] + (1-EMA_ALPHA)*ema_center[1])
            )

        # --- 이탈 판단 ---
        dx, dy = ema_center[0] - home[0], ema_center[1] - home[1]
        dist = math.hypot(dx, dy)
        deviated_now = (score < SCORE_THRESH) or (dist > POS_THRESH_PX)

        now = time.time()
        if deviated_now:
            deviate_run += 1
            if deviate_run >= DEVIATE_FRAMES_REQ and deviate_since is None:
                deviate_since = now
        else:
            deviate_run = 0
            deviate_since = None

        alert = False; remain = 0
        if deviate_since is not None:
            elapsed = now - deviate_since
            remain = max(0, int(GRACE_SECS - elapsed))
            if elapsed >= GRACE_SECS:
                alert = True

        # --- 시각화 ---
        vis = frame.copy()
        tw2, th2 = bm["size"]
        x,y = bm["top_left"]
        cv2.rectangle(vis, (x,y), (x+tw2, y+th2), (0,255,0), 2)
        cv2.circle(vis, home, 5, (255,255,0), -1)
        cv2.circle(vis, ema_center, 5, (0,0,255) if deviated_now else (0,255,0), -1)
        cv2.line(vis, home, ema_center, (0,0,255) if deviated_now else (0,255,0), 2)

        cv2.putText(vis, f"score={score:.2f} dist={int(dist)}px rot={bm['deg']} deg",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        cv2.putText(vis, f"POS_TH={POS_THRESH_PX}px  EMA={EMA_ALPHA}  N={DEVIATE_FRAMES_REQ}",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,255,180), 2)

        if deviate_since and not alert:
            cv2.putText(vis, f"Return within {remain}s",
                        (20,95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        if alert:
            cv2.putText(vis, "ALERT: Pattern deviated!",
                        (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        cv2.imshow("template_monitor (stabilized)", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
