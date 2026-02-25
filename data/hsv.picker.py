import cv2
import numpy as np

def nothing(x):
    pass

# 웹캠 연결
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 트랙바 윈도우 생성
cv2.namedWindow('Trackbars')
cv2.createTrackbar('L-H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('L-S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('L-V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('U-H', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('U-S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('U-V', 'Trackbars', 255, 255, nothing)

print("--- 사용법 ---")
print("1. 테이프가 화면에 나오게 하세요.")
print("2. 슬라이더를 조절해서 테이프 영역만 '흰색'이 되게 만드세요.")
print("3. 그때의 하단 출력값(L-H, L-S...)을 기록하세요.")

while True:
    ret, frame = cap.read()
    if not ret: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 트랙바에서 현재 값 읽기
    l_h = cv2.getTrackbarPos('L-H', 'Trackbars')
    l_s = cv2.getTrackbarPos('L-S', 'Trackbars')
    l_v = cv2.getTrackbarPos('L-V', 'Trackbars')
    u_h = cv2.getTrackbarPos('U-H', 'Trackbars')
    u_s = cv2.getTrackbarPos('U-S', 'Trackbars')
    u_v = cv2.getTrackbarPos('U-V', 'Trackbars')

    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([u_h, u_s, u_v])

    # 이진 분류 (마스킹)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask (Binary)', mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(1) == ord('q'):
        print(f"최종 선택 범위: Lower [{l_h}, {l_s}, {l_v}], Upper [{u_h}, {u_s}, {u_v}]")
        break

cap.release()
cv2.destroyAllWindows()