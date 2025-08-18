#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 — 템플릿 매칭(전역) 기본기
- 매 프레임에서 전체 화면에 대해 matchTemplate 실행
- 최고점 위치에 박스(초록) 표시 + 점수 출력
- 아직 로컬 검색/엣지/이탈 판정 없음 (기본 개념만 익히기)

실행 예:
  python step2_template_match_baseline.py --video chair_test.mov --template assets/chair_template.png

필요 패키지:
  pip install opencv-python
"""

import cv2
import argparse
from typing import Tuple


# ---------- 유틸: 가로폭 기준 리사이즈(표시/속도 조절용) ----------
def resize_keep_w(img, max_w: int) -> Tuple:
    """
    이미지를 '가로폭 기준'으로 리사이즈.
    - max_w가 0이거나 현재 가로가 더 작으면 원본 유지.
    - 반환: (리사이즈된_이미지, scale)
      * scale = 리사이즈_가로 / 원본_가로 (나중에 좌표 환산할 때 사용 가능)
    """
    h, w = img.shape[:2]
    if max_w and w > max_w:
        scale = max_w / float(w)
        new_size = (int(w * scale), int(h * scale))  # (width, height)
        return cv2.resize(img, new_size), scale
    return img, 1.0


def parse_args():
    p = argparse.ArgumentParser(description="Step 2 — 템플릿 매칭(전역) 기본기")
    p.add_argument("--video", type=str, default="../../assets/chair_test_480p_30fps.mp4",
                   help="영상 경로 또는 0(웹캠). 기본: ../../assets/chair_test_480p_30fps.mp4")
    p.add_argument("--template", type=str, default="../../assets/chair_template.png",
                   help="템플릿 이미지 경로")
    p.add_argument("--maxw", type=int, default=1280,
                   help="프레임 표시/처리용 최대 가로폭(속도/가독성 용). 기본: 1280")
    p.add_argument("--method", type=str, default="TM_CCOEFF_NORMED",
                   help="매칭 방법 (TM_SQDIFF, TM_CCORR, TM_CCOEFF, *_NORMED 등)")
    return p.parse_args()


def get_method(name: str) -> int:
    """
    문자열로 받은 매칭 방법명을 OpenCV 상수로 변환.
    - 기본/권장: TM_CCOEFF_NORMED (값이 클수록 유사)
    - TM_SQDIFF(_NORMED)는 '작을수록' 유사이니 해석 주의
    """
    table = {
        "TM_SQDIFF": cv2.TM_SQDIFF,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    }
    return table.get(name.upper(), cv2.TM_CCOEFF_NORMED)


def main():
    args = parse_args()

    # 1) 입력 열기
    source = 0 if str(args.video) == "0" else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[Error] Cannot open source: {args.video}")

    # 2) 템플릿 읽기
    tmpl_bgr = cv2.imread(args.template)
    if tmpl_bgr is None:
        cap.release()
        raise SystemExit(f"[Error] Cannot read template: {args.template}")

    # 3) 그레이스케일 변환(기본기: 컬러보다 그레이로 매칭이 일관적이고 빠름)
    tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]
    print(f"[Info] Template size: {tw}x{th}")

    # 4) 매칭 방법
    method = get_method(args.method)
    print(f"[Info] Matching method: {args.method}")

    # 5) 프레임 반복
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # (선택) 가로폭 기준 리사이즈: 속도와 보기 편의성
        frame_resized, scale = resize_keep_w(frame, args.maxw)

        # 프레임도 그레이로
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        # 템플릿이 프레임보다 크면 매칭 불가 → 처리 전에 템플릿 축소(학습단계에선 간단히 스킵)
        if th > H or tw > W:
            cv2.putText(frame_resized, "Template larger than frame — resize your template",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("template_match_baseline", frame_resized)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # 6) 템플릿 매칭 (전역) — 결과 행렬 res에서 최고/최저값/위치 추출
        res = cv2.matchTemplate(gray, tmpl_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 7) 방법별 점수 해석
        # - TM_SQDIFF(_NORMED): '작을수록' 유사 → min_val/min_loc 사용
        # - 그 외(CCORR, CCOEFF 등): '클수록' 유사 → max_val/max_loc 사용
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            score = 1.0 - float(min_val)  # 직관적 표시용(클수록 좋다로 보기 위해 변환)
            top_left = min_loc
        else:
            score = float(max_val)
            top_left = max_loc

        # 8) 박스 좌표 계산
        x1, y1 = top_left
        x2, y2 = x1 + tw, y1 + th

        # 9) 그리기(초록 박스 + 점수)
        vis = frame_resized.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"score={score:.3f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("template_match_baseline", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()