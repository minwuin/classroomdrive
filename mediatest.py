import cv2
import mediapipe as mp
import os

# 1. MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# [디테일] 발표 자료용으로 선 굵기와 색상을 눈에 띄게 커스텀 (형광 녹색/빨간색)
custom_landmark_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4)  # 관절점(빨강)
custom_connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2) # 연결선(녹색)

# 2. 저장 폴더 확인
save_dir = 'presentation_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 3. 웹캠 세팅
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

win_name = 'MediaPipe Full Skeleton Viewer'
cv2.namedWindow(win_name)

print("\n=== [전신 관절 캡처 모드] ===")
print("1. 카메라 앞에서 전신(또는 상반신)이 잘 나오게 서보세요.")
print("2. 's'를 누르면 현재 관절이 그려진 화면이 고화질로 저장됩니다.")
print("3. 'q'를 누르면 종료됩니다.\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    # MediaPipe는 RGB 이미지를 사용하므로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # 안내 문구 추가 (저장되는 사진에는 안 나오게 복사본 사용)
    display_frame = frame.copy()

    # 관절이 인식되었다면 화면에 그리기
    if results.pose_landmarks:
        # 화면에 33개의 모든 관절과 뼈대를 그려줍니다.
        mp_drawing.draw_landmarks(
            display_frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=custom_landmark_style,
            connection_drawing_spec=custom_connection_style
        )
        
        # 실제 저장용 원본 frame에도 똑같이 그려줍니다 (안내 문구 없이 저장하기 위함)
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=custom_landmark_style,
            connection_drawing_spec=custom_connection_style
        )

    # 화면 UI 문구
    cv2.putText(display_frame, "Stand in front of the camera", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_frame, "Press 's' to Save / 'q' to Quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow(win_name, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_path = os.path.join(save_dir, 'skeleton_view.jpg')
        cv2.imwrite(save_path, frame) # 안내 문구 없는 깨끗한 frame 저장
        print(f"📸 관절 인식 사진 저장 완료! ({save_path})")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()