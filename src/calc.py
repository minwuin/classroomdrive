import cv2
import numpy as np
import os

# [설정] 실제 거리값 (utils.py와 동일하게 맞추세요)
REAL_WIDTH = 0.210
REAL_HEIGHT = 3.280

pts = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        print(f"점 추가됨 ({len(pts)}/4): {x, y}")

# 저장 폴더 생성
if not os.path.exists('data'):
    os.makedirs('data')

cap = cv2.VideoCapture(0)
# 웹캠 해상도를 HD급으로 키워 정밀도를 높입니다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

win_name = 'Calibration'
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(win_name, mouse_callback)

print("--- 사용법 ---")
print("1. 바닥의 사각형 영역 4곳을 클릭하세요.")
print("2. 순서: 좌측상단 -> 우측상단 -> 우측하단 -> 좌측하단")
print("3. 4개를 다 찍으면 자동으로 행렬이 저장되고 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 찍은 점 표시
    for p in pts:
        cv2.circle(frame, (p[0], p[1]), 5, (0, 255, 0), -1)

    cv2.imshow('Calibration', frame)

    # 4개 점이 모두 찍혔을 때 처리
    if len(pts) == 4:
        # 1. 실제 거리 행렬 계산 (meters 단위)
        src_points = np.float32(pts)
        dst_points_real = np.float32([
            [0, 0], [REAL_WIDTH, 0], 
            [REAL_WIDTH, REAL_HEIGHT], [0, REAL_HEIGHT]
        ])
        
        # 실제 거리 변환 행렬 M 생성 및 저장
        M = cv2.getPerspectiveTransform(src_points, dst_points_real)
        np.save('data/M.npy', M)
        
        # 2. 시각화용(Bird's Eye View) 행렬 생성
        dst_points_view = np.float32([[0, 0], [400, 0], [400, 600], [0, 600]])
        M_view = cv2.getPerspectiveTransform(src_points, dst_points_view)
        np.save('data/image_mat.npy', M_view)
        
        print("행렬 저장 완료! data 폴더를 확인하세요.")
        break

    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()