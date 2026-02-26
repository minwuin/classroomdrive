import cv2
import numpy as np
import os

# 1. 탑뷰 변환 행렬 로드
try:
    M_view = np.load('data/image_mat.npy')
    print("성공: 변환 행렬(image_mat.npy)을 불러왔습니다.")
except FileNotFoundError:
    print("에러: 'data/image_mat.npy' 파일이 없습니다. 01_calibration.py를 먼저 실행하세요.")
    exit()

# 저장 폴더 생성
save_dir = 'presentation_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 2. 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# [수정 1] 전체화면 강제 고정 해제 (비율 깨짐 방지)
win_name = 'Presentation Capture Mode'
cv2.namedWindow(win_name) 

# ==========================================================
# [수정 2] 탑뷰 줌아웃(Zoom-out) 행렬 적용
# 4개의 점 바깥쪽 공간까지 넓게 보여주기 위한 마법의 행렬입니다.
# ==========================================================
zoom_scale = 0.5  # 0.5 = 2배 넓게 보기 (원하면 0.3이나 0.8로 조절 가능)
tx = 400 * (1 - zoom_scale) / 2
ty = 600 * (1 - zoom_scale) / 2

zoom_matrix = np.array([
    [zoom_scale, 0, tx],
    [0, zoom_scale, ty],
    [0, 0, 1]
], dtype=np.float32)

# 기존 행렬에 줌아웃 효과 곱하기
M_view_zoomed = np.dot(zoom_matrix, M_view)
# ==========================================================

print("\n=== [사용 방법] ===")
print("1. [원본 화면] 피사체를 배치하고 's'를 누르면 원본이 저장되고 탑뷰로 넘어갑니다.")
print("2. [탑뷰 화면] 탑뷰 상태를 확인하고 's'를 누르면 탑뷰가 저장되고 프로그램이 종료됩니다.")
print("3. 언제든 'q'를 누르면 캡처를 취소하고 종료할 수 있습니다.\n")

view_state = 'original' 

while True:
    ret, frame = cap.read()
    if not ret: break

    display_frame = frame.copy()

    if view_state == 'original':
        cv2.putText(display_frame, "[1] ORIGINAL VIEW - Press 's' to save", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(win_name, display_frame)
        
    elif view_state == 'bev':
        # [핵심] 기존 M_view 대신 줌아웃이 적용된 M_view_zoomed 사용!
        bev_frame = cv2.warpPerspective(frame, M_view_zoomed, (400, 600))
        display_bev = bev_frame.copy()
        cv2.putText(display_bev, "[2] BIRD'S EYE VIEW - Press 's' to save", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win_name, display_bev)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        if view_state == 'original':
            orig_path = os.path.join(save_dir, 'original_view.jpg')
            cv2.imwrite(orig_path, frame)
            print(f"📸 원본 사진 저장 완료: {orig_path}")
            view_state = 'bev'
            
        elif view_state == 'bev':
            bev_path = os.path.join(save_dir, 'birds_eye_view.jpg')
            # 글씨 없는 깨끗한 탑뷰 저장
            cv2.imwrite(bev_path, bev_frame)
            print(f"📸 탑뷰 사진 저장 완료: {bev_path}")
            print("\n✅ 두 장의 사진이 모두 성공적으로 저장되었습니다! 프로그램을 종료합니다.")
            break
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()