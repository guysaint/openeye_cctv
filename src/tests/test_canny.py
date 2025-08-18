#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 — 첫 프레임에서 ROI(의자) 선택 후 템플릿으로 저장하는 스크립트
- 한 단계 학습용: 엣지/매칭/로컬검색 없음.
- 마우스로 ROI를 드래그하고 ENTER/SPACE로 확정(C 키로 취소).

실행 예:
  python roi_canny.py --video chair_test_480p_30fps.mp4 --out-dir img
  python roi_canny.py --video 0 --out-dir img   # 웹캠 사용

필요 패키지:
  pip install opencv-python
  # 만약 selectROI가 없다면:
  pip install opencv-contrib-python
"""

import os
import json
import argparse
from typing import Tuple
import cv2


# ---------- 유틸: 미리보기(표시용)로 가로폭 제한 리사이즈 ----------
def resize_keep_w(img, max_w: int) -> Tuple:
    """
    이미지를 '가로폭 기준'으로 리사이즈.
    - max_w가 0이거나 현재 가로폭이 더 작으면 원본 유지.
    - 반환값: (리사이즈된_이미지, scale)
        * scale = (리사이즈_가로 / 원본_가로)
        * 나중에 미리보기 좌표를 원본 좌표로 환산할 때 사용.
    """
    h, w = img.shape[:2]
    if max_w and w > max_w:
        scale = max_w / float(w)
        new_size = (int(w * scale), int(h * scale))  # (width, height)
        return cv2.resize(img, new_size), scale
    return img, 1.0


# ---------- 인자 파싱 ----------
def parse_args():
    p = argparse.ArgumentParser(description="Step 1 — 첫 프레임에서 ROI 선택 후 템플릿 저장")
    p.add_argument("--video", type=str, default="../img/chair_test_480p_30fps.mp4",
                   help="영상 경로 또는 0(웹캠). 기본: chair_test_480p_30fps.mp4")
    p.add_argument("--out-dir", type=str, default="../img",
                   help="템플릿을 저장할 폴더. 기본: img")
    p.add_argument("--out-name", type=str, default="../img/chair_template.png",
                   help="템플릿 파일명. 기본: chair_template.png")
    p.add_argument("--maxw", type=int, default=1280,
                   help="미리보기 최대 가로폭(표시 전용 리사이즈). 기본: 1280")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) 영상/카메라 열기
    #    - 문자열 "0"이 넘어오면 실제 장치 인덱스 0으로 변환
    source = 0 if str(args.video) == "0" else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[Error] Cannot open source: {args.video}")

    # 2) 첫 프레임 읽기
    #    - 실패하면 더 진행하지 않고 종료(문제 원인부터 해결)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise SystemExit("[Error] Failed to read the first frame")

    # 3) 미리보기 이미지 만들기(표시만 축소; 분석/저장은 원본 좌표 사용)
    preview, scale = resize_keep_w(frame, args.maxw)

    # 4) ROI 선택 UI 열기
    #    - 마우스로 드래그 → ENTER/SPACE 확정, C 취소
    win_title = "Select ROI (ENTER confirm, C cancel)"
    x, y, w_roi, h_roi = cv2.selectROI(win_title, preview,
                                       fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win_title)  # 선택 창 닫기

    # 5) 취소 처리(폭/높이 0이면 취소로 간주)
    if w_roi == 0 or h_roi == 0:
        cap.release()
        print("[Info] ROI selection canceled.")
        return

    # 6) 미리보기 좌표 → 원본 좌표로 환산
    #    - preview는 scale배만큼 줄어든 상태이므로 /scale
    x0 = int(x / scale)
    y0 = int(y / scale)
    w0 = int(w_roi / scale)
    h0 = int(h_roi / scale)

    # 7) 원본 프레임에서 템플릿 잘라내기
    template = frame[y0:y0 + h0, x0:x0 + w0].copy()

    # 8) 시각 확인(원본 프레임에 박스, 템플릿 창)
    vis = frame.copy()
    cv2.rectangle(vis, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)  # 초록 사각형
    cv2.imshow("First frame + ROI", vis)
    cv2.imshow("Template (chair)", template)
    print(f"[Info] ROI on original frame: x={x0}, y={y0}, w={w0}, h={h0}")
    cv2.waitKey(0)  # 아무 키 누르면 진행

    # 9) 템플릿/메타데이터 저장
    os.makedirs(args.out_dir, exist_ok=True)
    out_img = os.path.join(args.out_dir, args.out_name)
    ok = cv2.imwrite(out_img, template)
    if not ok:
        cap.release()
        raise SystemExit(f"[Error] Failed to save template image to {out_img}")

    meta = {
        "video": args.video,                 # 어떤 입력에서 땄는지
        "frame_shape_hw": [int(frame.shape[0]), int(frame.shape[1])],  # [H, W]
        "preview_scale": scale,              # 미리보기 축소 배율(디버그 참고용)
        "roi_xywh": [x0, y0, w0, h0],        # 원본 좌표의 템플릿 박스
        "template_path": out_img,            # 저장된 템플릿 경로
    }
    meta_path = os.path.join(
        args.out_dir,
        os.path.splitext(args.out_name)[0] + "_meta.json"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Template saved to: {out_img}")
    print(f"[OK] Metadata saved to: {meta_path}")

    # 10) 자원 해제
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()