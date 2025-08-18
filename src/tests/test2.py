import cv2
import numpy as np

img = cv2.imread("../../assets/chair_template.png")

# @케니 엣지 적용 
edges = cv2.Canny(img,100,200)

# @이미지 출력
cv2.imshow('Original', img)
cv2.imshow('Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()