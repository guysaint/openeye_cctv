import cv2, numpy as np, time, math
from typing import Tuple

# ===== 사용자 설정 =====
SOURCE = "../../assets/chair_test_480p_30fps.mp4"  # 0 이면 웹캠
TEMPLATE_PATH = "../../assets/chair_template.png"  # 기준 템플릿 파일
FRAME_MAX_W = 720     # 입력 프레임을 이 너비로 축소 (속도 ↑), 0이면 원본
ANALYZE_EVERY = 2      # N프레임마다 분석(1=매프레임, 2=절반)
POS_THRESH_PX = 100     # 기준 위치로부터 허용 편차(픽셀, 축소 후 기준)
SCORE_THRESH  = 0.8   # 유사도 임계 (TM_CCOEFF_NORMED: 0~1, 높을수록 유사)
GRACE_SECS    = 10     # 이탈 지속 허용 시간
ROT_DEGS      = list(range(-15, 16, 5)) # 템플릿 회전 각도 스윕(도 단위)
METHOD        = cv2.TM_CCOEFF_NORMED     # 정규화 상관

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
    # 조명 변화에 조금 더 강하게: 그레이+약블러+엣지(선택)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return g

def best_match(frame_gray, templates_gray):
    # 여러 회전 템플릿 중 최고 유사/위치/각도 찾기
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

def main():
    # 템플릿 로드 & 축소 & 회전군 준비
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        raise SystemExit(f"Cannot read template: {TEMPLATE_PATH}")
    template, _ = resize_keep_w(template, 300)  # 템플릿도 과소/과대 방지
    templates_gray = [(deg, preprocess_for_matching(rotate_img(template, deg)))
                      for deg in ROT_DEGS]

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {SOURCE}")

    # 첫 프레임에서 기준 위치/각도/점수 계산
    ok, first = cap.read()
    if not ok: 
        raise SystemExit("No first frame")
    first, scale_r = resize_keep_w(first, FRAME_MAX_W)
    first_g = preprocess_for_matching(first)
    bm0 = best_match(first_g, templates_gray)
    home = center_from(bm0["top_left"], bm0["size"])
    home_deg = bm0["deg"]
    print(f"[INIT] score={bm0['score']:.3f}, home={home}, deg={home_deg}")

    deviate_since = None
    fcount = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        fcount += 1
        if fcount % ANALYZE_EVERY != 0:
            # 디스플레이만 계속할 수도 있지만 여기선 스킵
            continue

        frame, _ = resize_keep_w(frame, FRAME_MAX_W)
        fg = preprocess_for_matching(frame)
        bm = best_match(fg, templates_gray)
        c  = center_from(bm["top_left"], bm["size"])
        dx, dy = c[0]-home[0], c[1]-home[1]
        dist = math.hypot(dx, dy)
        score = bm["score"]

        # 이탈 조건: 점수 낮거나(매칭 불가) +/or 위치 크게 변함
        # 점수 기준이 너무 타이트하면 미탐, 너무 느슨하면 오탐 → 0.5~0.7 구간에서 튜닝
        deviated = (score < SCORE_THRESH) or (dist > POS_THRESH_PX)

        now = time.time()
        if deviated:
            if deviate_since is None:
                deviate_since = now
        else:
            deviate_since = None

        alert = False; remain = 0
        if deviate_since is not None:
            elapsed = now - deviate_since
            remain = max(0, int(GRACE_SECS - elapsed))
            if elapsed >= GRACE_SECS:
                alert = True

        # 시각화
        vis = frame.copy()
        tw, th = bm["size"]
        x,y = bm["top_left"]
        cv2.rectangle(vis, (x,y), (x+tw, y+th), (0,255,0), 2)
        cv2.circle(vis, home,   5, (255,255,0), -1)
        cv2.circle(vis, c,      5, (0,0,255) if deviated else (0,255,0), -1)
        cv2.line(vis, home, c, (0,0,255) if deviated else (0,255,0), 2)
        cv2.putText(vis, f"score={score:.2f} dist={int(dist)}px rot={bm['deg']}deg",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if deviate_since and not alert:
            cv2.putText(vis, f"Return within {remain}s",
                        (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        if alert:
            cv2.putText(vis, "ALERT: Pattern deviated!",
                        (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        cv2.imshow("template_monitor", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()