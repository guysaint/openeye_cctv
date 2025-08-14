# file: src/chair_template_monitor.py
import cv2, time, numpy as np, math

SOURCE = '../../assets/chair_test_480p_30fps.mp4'  # 0=웹캠, 또는 "assets/your_video.mp4"
POS_THRESH_PX = 30     # 원위치로부터 허용 편차(픽셀)
GRACE_SECS = 10        # 편차가 임계 이상 지속되면 경고
METHOD = cv2.TM_CCOEFF_NORMED  # 템플릿 매칭 방법

# --- 템플릿 선택용 콜백 ---
sel = {"drag": False, "x1":0,"y1":0,"x2":0,"y2":0, "done":False}
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        sel["drag"] = True; sel["x1"], sel["y1"] = x, y
    elif event == cv2.EVENT_MOUSEMOVE and sel["drag"]:
        sel["x2"], sel["y2"] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        sel["drag"] = False; sel["x2"], sel["y2"] = x, y; sel["done"] = True

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open source: {SOURCE}")

# --- 첫 프레임에서 템플릿 선택 ---
ok, first = cap.read()
if not ok: raise RuntimeError("Cannot read first frame")

clone = first.copy()
cv2.namedWindow("select_template")
cv2.setMouseCallback("select_template", on_mouse)

while True:
    disp = clone.copy()
    if sel["drag"] or sel["done"]:
        x1,y1,x2,y2 = sel["x1"],sel["y1"],sel["x2"],sel["y2"]
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(disp, "Drag a box around the CHAIR, then press ENTER",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("select_template", disp)
    k = cv2.waitKey(10)
    if k == 13 and sel["done"]:  # Enter
        break
    if k == 27:  # ESC
        cap.release(); cv2.destroyAllWindows(); raise SystemExit("Canceled")

cv2.destroyWindow("select_template")
x1,y1,x2,y2 = sel["x1"],sel["y1"],sel["x2"],sel["y2"]
x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
template = first[y1:y2, x1:x2].copy()
th, tw = template.shape[:2]

# 초기 위치(원위치) 계산
res0 = cv2.matchTemplate(first, template, METHOD)
_, maxv0, _, maxloc0 = cv2.minMaxLoc(res0)
home = (maxloc0[0] + tw//2, maxloc0[1] + th//2)

deviate_since = None
while True:
    ok, frame = cap.read()
    if not ok: time.sleep(0.01); continue

    # 템플릿 매칭
    res = cv2.matchTemplate(frame, template, METHOD)
    minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
    # TM_CCOEFF_NORMED는 maxv가 클수록 좋음
    top_left = maxloc
    center = (top_left[0] + tw//2, top_left[1] + th//2)

    # 거리 계산
    dx = center[0] - home[0]
    dy = center[1] - home[1]
    dist = math.hypot(dx, dy)

    # 이탈 판정 (신뢰도 보정: 매칭 점수 너무 낮으면 스킵)
    # 필요시 threshold 조정 (예: 0.6)
    confidence_ok = (maxv > 0.5)
    deviated = confidence_ok and (dist > POS_THRESH_PX)

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
    cv2.rectangle(vis, top_left, (top_left[0]+tw, top_left[1]+th), (0,255,0), 2)
    cv2.circle(vis, home, 5, (255,255,0), -1)
    cv2.circle(vis, center, 5, (0,0,255) if deviated else (0,255,0), -1)
    cv2.line(vis, home, center, (0,0,255) if deviated else (0,255,0), 2)

    cv2.putText(vis, f"match={maxv:.2f} dist={dist:.0f}px", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    if deviate_since and not alert:
        cv2.putText(vis, f"Return within {remain}s", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    if alert:
        cv2.putText(vis, "ALERT: Chair not back to home!", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

    cv2.imshow("chair_template_monitor", vis)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release(); cv2.destroyAllWindows()