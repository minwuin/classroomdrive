import cv2
import mediapipe as mp
import numpy as np
import time
from utils import AutonomousUtils

# 1. 초기화
utils = AutonomousUtils()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 변환 행렬 로드 (시각화 및 거리 계산용)
M_view = np.load('data/image_mat.npy')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 전체화면 설정
win_name = 'KDT Autonomous Driving System'
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- [추가] 옵티컬 플로우(내 속도 측정) 파라미터 ---
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

old_gray = None
p0 = None
ego_speed = 0.0 # 내 차량 속도 초기화
# ---------------------------------------------

# 변수 초기화
prev_pos_m = None 
prev_loop_time = time.time() # 루프 전체 시간 측정용으로 이름 변경

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # --- [추가/수정] 전체 프레임 dt 계산 ---
    current_time = time.time()
    dt = current_time - prev_loop_time
    prev_loop_time = current_time
    
    h_img, w_img = frame.shape[:2]

    # ==========================================================
    # --- [기능 5: 광학 흐름(Optical Flow)으로 내 속도 구하기] ---
    # ==========================================================
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if old_gray is None:
        old_gray = gray_frame
        # 바닥 부분(화면 하단 50%)에서만 점을 찾도록 마스크 씌우기
        mask = np.zeros_like(old_gray)
        mask[int(h_img*0.5):, :] = 255
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
    else:
        if p0 is not None and len(p0) > 0:
            # 점 추적
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            speeds = []
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                
                ## 픽셀 좌표를 실제 미터(m)로 변환
                m_new_x, m_new_y = utils.pixel_to_meter(a, b)
                m_old_x, m_old_y = utils.pixel_to_meter(c, d)
                
                # [핵심 수정] X축(좌우)과 Y축(앞뒤) 이동량을 분리
                dist_x = abs(m_new_x - m_old_x)
                dist_y = abs(m_new_y - m_old_y)
                
                # 회전 노이즈 필터링: 
                # 프레임당 좌우(X축)로 너무 심하게 튀는 점은 카메라 회전으로 간주하고 무시함
                if dist_x < 0.05: 
                    # 정상적인 점이라면 오직 '앞뒤(Y축)' 이동 거리만 속도 계산에 사용
                    speeds.append(dist_y)
                
                # 추적하는 점을 파란색으로 화면에 표시
                cv2.circle(frame, (int(a), int(b)), 3, (255, 0, 0), -1)

            if len(speeds) > 0 and dt > 0:
                # 점들의 평균 이동 속도
                current_speed = np.mean(speeds) / dt
                
                if current_speed < 0.02: current_speed = 0.0 # 데드존
                # 내 속도에 EMA 필터 적용 (조금 묵직하게 alpha=0.1)
                ego_speed = (0.1 * current_speed) + (0.9 * ego_speed)

            # 점이 10개 미만으로 줄면 새로 추출
            p0 = good_new.reshape(-1, 1, 2)
            if len(p0) < 10:
                mask = np.zeros_like(gray_frame)
                mask[int(h_img*0.5):, :] = 255
                p0 = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
        else:
            mask = np.zeros_like(gray_frame)
            mask[int(h_img*0.5):, :] = 255
            p0 = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
            
        old_gray = gray_frame.copy()
    
    # --- [기능 1: Bird's Eye View 생성] ---
    bev_frame = cv2.warpPerspective(frame, M_view, (400, 600))
    
    # --- [기능 2: 보행자 인식 및 물리량 계산] ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    status_msg = "DRIVE SAFE"
    ui_color = (0, 255, 0) # 기본 초록색 바

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        px = int((lm[mp_pose.PoseLandmark.LEFT_ANKLE].x + lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x) / 2 * w_img)
        py = int((lm[mp_pose.PoseLandmark.LEFT_ANKLE].y + lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2 * h_img)
        
        # 1. 현재의 Raw(날것) 좌표 변환
        curr_x_m, curr_y_m = utils.pixel_to_meter(px, py)
        
        # 2. EMA 필터 및 속도 계산 로직 적용
        if prev_pos_m is not None:
            if dt > 0:
                # -------------------------------------------------------------
                # [EMA 필터] 이전 좌표와 현재 좌표를 섞어 노이즈 제거
                # alpha가 작을수록(0.2) 움직임이 묵직하고 부드러워짐
                alpha = 0.2 
                curr_x_m = (alpha * curr_x_m) + ((1 - alpha) * prev_pos_m[0])
                curr_y_m = (alpha * curr_y_m) + ((1 - alpha) * prev_pos_m[1])
                # -------------------------------------------------------------

                # 필터링된 좌표 기준으로 이동 거리 및 속도 계산
                dist_moved = np.sqrt((curr_x_m - prev_pos_m[0])**2 + (curr_y_m - prev_pos_m[1])**2)
                total_velocity = dist_moved / dt 
                velocity_y = (curr_y_m - prev_pos_m[1]) / dt
                
                # -------------------------------------------------------------
                # [데드존] 미세한 떨림(0.02m/s 미만)은 속도를 0으로 강제 고정
                if total_velocity < 0.02: 
                    total_velocity = 0.0
                if abs(velocity_y) < 0.02: 
                    velocity_y = 0.0
                # -------------------------------------------------------------
                
                # 판단 및 계산 함수 호출
                status, box_color = utils.get_behavior_status(total_velocity)
                ttc = utils.calculate_ttc(curr_y_m, velocity_y) 

                # 시각화 (UI 출력)
                label = f"{status} | Dist: {curr_y_m:.1f}m | Vel: {total_velocity:.2f}m/s"
                
                if ttc < 5.0: # 5초 이내로 가까워질 때만 경고
                    status_msg = f"!!! COLLISION ALERT: {ttc:.1f}s !!!"
                    ui_color = (0, 0, 255)
                else:
                    status_msg = "DRIVE SAFE"
                    ui_color = (0, 255, 0)
                
                cv2.rectangle(frame, (px-60, py-120), (px+60, py), box_color, 3)
                cv2.putText(frame, label, (px-110, py-130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # 다음 프레임 계산을 위해 '현재 값(필터링된 값)'을 '이전 값'으로 저장
        prev_pos_m = (curr_x_m, curr_y_m)
        prev_time = current_time

    # --- [기능 3: 차선 감지 및 가이드 로직] ---
    # 화면 하단 40% 영역을 차선 인식 ROI로 설정
    roi_h = int(h_img * 0.4)
    roi = frame[h_img - roi_h:, :]
    
    # utils에서 정의한 HSV 필터로 차선 중심 찾기
    lane_cx, lane_mask = utils.get_lane_center(roi)
    
    if lane_cx is not None:
        # 차선 이탈 안내 메시지 생성
        lane_guide, _ = utils.get_lane_guide(lane_cx, w_img)
        
        # [수정된 부분] -------------------------------------------------------
        # 1. 감지된 차선 중심 그리기 (노란색 직선)
        # ROI 영역의 위쪽 끝(h_img - roi_h)부터 화면 맨 아래(h_img)까지 수직선 그리기
        # 색상: (0, 255, 255) -> 노란색 (Blue=0, Green=255, Red=255)
        cv2.line(frame, (lane_cx, h_img - roi_h), (lane_cx, h_img), (0, 255, 0), 3)
        
        # 2. 내 차량의 기준 중심 그리기 (초록색 직선 - 고정)
        # 색상: (0, 255, 0) -> 초록색
        cv2.line(frame, (w_img//2, h_img - roi_h), (w_img//2, h_img), (0, 255, 255), 2)

    else:
        lane_guide = "LANE LOST"

    # --- [기능 4: UI 렌더링] ---
    # 상단 상태바
    cv2.rectangle(frame, (0, 0), (w_img, 80), (30, 30, 30), -1) # 기본 어두운 배경
    
    # 1. 좌측 영역: 차선 및 주행 상태 (LANE)
    # 차선 상태에 따른 텍스트 색상 변경
    lane_text_color = (0, 255, 0) if "CENTER" in lane_guide else (0, 165, 255)
    if lane_guide == "LANE LOST": lane_text_color = (128, 128, 128)
    
    cv2.putText(frame, "LANE CONTROL", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, lane_guide, (20, 65), cv2.FONT_HERSHEY_DUPLEX, 1.2, lane_text_color, 2)

    # 2. 중앙 구분선
    cv2.line(frame, (w_img//2, 10), (w_img//2, 70), (100, 100, 100), 2)

    # 3. 우측 영역: 충돌 알림 (SAFETY)
    # 충돌 위험 시 우측 바 배경을 빨간색으로 강조 
    if ui_color == (0, 0, 255): # 경고 상태일 때
        cv2.rectangle(frame, (w_img//2, 0), (w_img, 80), (0, 0, 255), -1)
        alert_text_color = (255, 255, 255)
    else:
        alert_text_color = (0, 255, 0)

    cv2.putText(frame, "SAFETY ALERT", (w_img//2 + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, status_msg, (w_img//2 + 20, 65), cv2.FONT_HERSHEY_DUPLEX, 1.2, alert_text_color, 2)

    # --- [수정] 내 차량(노트북) 속도 출력 (우측 하단) ---
    ego_text = f"EGO SPEED: {ego_speed:.2f} m/s"
    
    cv2.rectangle(frame, (w_img - 380, h_img - 60), (w_img, h_img), (20, 20, 20), -1)
    cv2.putText(frame, ego_text, (w_img - 350, h_img - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 200, 0), 2)

    # 전체 화면 출력 (한 번만!)
    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()