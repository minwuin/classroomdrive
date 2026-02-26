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

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    ped_bbox = None  # 보행자 바운딩 박스 좌표 (x1, y1, x2, y2)
    px, py = 0, 0


    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        px = int((lm[mp_pose.PoseLandmark.LEFT_ANKLE].x + lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x) / 2 * w_img)
        py = int((lm[mp_pose.PoseLandmark.LEFT_ANKLE].y + lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2 * h_img)
        
        # 보행자 주변 영역을 넉넉하게 박스로 설정 (Optical Flow 노이즈 방지용)
        ped_bbox = (px - 80, py - 150, px + 80, py + 30)

    
    # ==========================================================
    # --- [기능 5: 광학 흐름(Optical Flow)으로 내 속도 구하기] ---
    # ==========================================================
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def get_flow_mask():
        m = np.zeros_like(gray_frame)
        m[int(h_img*0.5):, :] = 255
        if ped_bbox is not None:
            x1, y1, x2, y2 = ped_bbox
            # 화면 밖으로 나가지 않게 제한
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            cv2.rectangle(m, (x1, y1), (x2, y2), 0, -1) # 보행자 영역은 검은색(0)으로 칠해 점 추출 방지
        return m
    
    if old_gray is None:
        old_gray = gray_frame
        mask = get_flow_mask()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
    else:
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            speeds = []
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                
                # [핵심] 이미 추적 중인 점이라도 보행자 영역 안으로 들어오면 계산에서 무시!
                if ped_bbox is not None:
                    x1, y1, x2, y2 = ped_bbox
                    if x1 <= a <= x2 and y1 <= b <= y2:
                        continue 
                
                c, d = old.ravel()
                m_new_x, m_new_y = utils.pixel_to_meter(a, b)
                m_old_x, m_old_y = utils.pixel_to_meter(c, d)
                
                dist_x = abs(m_new_x - m_old_x)
                dist_y = abs(m_new_y - m_old_y)
                
                if dist_x < 0.05: 
                    speeds.append(dist_y)
                cv2.circle(frame, (int(a), int(b)), 3, (255, 0, 0), -1)

            if len(speeds) > 0 and dt > 0:
                current_speed = np.mean(speeds) / dt
                if current_speed < 0.02: current_speed = 0.0 
                ego_speed = (0.1 * current_speed) + (0.9 * ego_speed)

            p0 = good_new.reshape(-1, 1, 2)
            if len(p0) < 10:
                mask = get_flow_mask()
                p0 = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
        else:
            mask = get_flow_mask()
            p0 = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
            
        old_gray = gray_frame.copy()
    
    # --- [기능 1: Bird's Eye View 생성] ---
    bev_frame = cv2.warpPerspective(frame, M_view, (400, 600))
    
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

                # 1. 카메라에 측정된 상대 속도 (나와 보행자가 가까워지는 속도)
                v_rel_x = (curr_x_m - prev_pos_m[0]) / dt
                v_rel_y = (curr_y_m - prev_pos_m[1]) / dt
                
                # 2. 보행자 절대 속도 보정 (내 차량의 속도를 빼서 보행자의 진짜 움직임만 추출)
                v_abs_y = v_rel_y - ego_speed
                
                # 피타고라스 정리로 보행자의 진짜 속도(절대 속도) 계산
                absolute_velocity = np.sqrt(v_rel_x**2 + v_abs_y**2)
                
                # -------------------------------------------------------------
                # [데드존] 미세한 떨림(0.02m/s 미만)은 속도를 0으로 강제 고정
                if absolute_velocity < 0.02: 
                    absolute_velocity = 0.0
                if abs(v_rel_y) < 0.02: 
                    v_rel_y = 0.0
                # -------------------------------------------------------------
                
                # 판단 및 계산 함수 호출                # 보행자의 상태(서있음/뜀)는 '절대 속도'로 판단하여 색상 결정
                status, box_color = utils.get_behavior_status(absolute_velocity)
                
                cam_x_m, cam_y_m = utils.pixel_to_meter(w_img // 2, h_img)
                
                # [추가] 카메라(나)와 보행자 사이의 진짜 떨어진 거리 계산
                actual_distance = cam_y_m - curr_y_m
                if actual_distance < 0: actual_distance = 0.0 # 내 뒤로 넘어가면 0으로 처리
                
                # 충돌 시간(TTC) 계산
                ttc = utils.calculate_ttc(curr_y_m, v_rel_y, cam_y_m)
                # -------------------------

                # 시각화 (UI 출력) - curr_y_m 대신 actual_distance 출력!
                label = f"{status} | Dist: {actual_distance:.1f}m | Vel: {absolute_velocity:.2f}m/s"
                
                
                if ttc < 3.0: # 5초 이내로 가까워질 때만 경고
                    status_msg = f"!!! COLLISION ALERT: {ttc:.1f}s !!!"
                    ui_color = (0, 0, 255)
                else:
                    status_msg = "DRIVE SAFE"
                    ui_color = (0, 255, 0)
                
                cv2.rectangle(frame, (px-60, py-120), (px+60, py), box_color, 3)
                cv2.putText(frame, label, (px-110, py-130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # 다음 프레임 계산을 위해 '현재 값(필터링된 값)'을 '이전 값'으로 저장
        prev_pos_m = (curr_x_m, curr_y_m)

    else:
        prev_pos_m = None

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