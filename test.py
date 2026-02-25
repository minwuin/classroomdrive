import cv2
import mediapipe as mp
import time

# 1. MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. 웹캠 연결 및 해상도 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # HD 해상도 가로
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # HD 해상도 세로

# 3. 전체화면 창 설정
win_name = 'Pedestrian Speed Test'
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 속도 계산을 위한 변수 초기화
prev_y = None
prev_time = None

while True:
    ret, frame = cap.read()
    if not ret: break
    
    current_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    status_msg = "Searching..."
    color = (0, 255, 0) # 기본 초록색 (Safe)

    if results.pose_landmarks:
        # 1. 발 좌표 추출 (왼쪽/오른쪽 발목의 평균 y값)
        landmarks = results.pose_landmarks.landmark
        left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
        right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
        curr_y = (left_foot_y + right_foot_y) / 2
        
        # 2. 속도 및 TTC 계산
        if prev_y is not None:
            dt = current_time - prev_time
            dy = curr_y - prev_y
            velocity = dy / dt # 초당 변위
            
            # 충돌 예상 시간 (TTC) 계산
            if velocity > 0.01: # 다가오는 중
                remaining_dist = 1.0 - curr_y
                ttc = remaining_dist / velocity
                
                status_msg = f"TTC: {ttc:.1f}s | Vel: {velocity:.2f}"
                
                # 3. 위험 알림 (TTC 2초 미만)
                if ttc < 2.0:
                    status_msg = f"!!! COLLISION WARNING: {ttc:.1f}s !!!"
                    color = (0, 0, 255) # 위험: 빨간색
            else:
                status_msg = "Safe / Moving Away"

        prev_y = curr_y
        prev_time = current_time
        
        # 사람 몸체 시각화 (랜드마크 연결)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # UI 출력 (상단 상태바)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), color, -1)
    # 전체화면이므로 글자 크기를 키웠습니다 (1.2 -> 1.5)
    cv2.putText(frame, status_msg, (40, 55), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.imshow(win_name, frame)
    
    # 'q'를 눌러 종료
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()