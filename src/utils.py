import cv2
import numpy as np
import time

# ==========================================
# [고정 변수 설정 영역] - 장소에 따라 이 값들을 수정하세요
# ==========================================
# 1. 실제 거리 설정 (01_calibration.py에서 설정한 사각형의 실제 크기)
REAL_WIDTH = 0.210
REAL_HEIGHT = 3.280

# 2. 속도 분류 임계값 (m/s 단위)
VEL_STANDING = 1  # 이 값보다 작으면 정지상태로 간주
VEL_WALKING = 2   # 이 값보다 크면 뛰는 상태로 간주

# 3. 차선 가이드 설정 (화면 가로 비율 기준)
LANE_HSV_LOWER = [0, 102, 108]
LANE_HSV_UPPER = [97, 255, 255]
LANE_CENTER_OFFSET = 0.1  # 중심에서 10% 이상 벗어나면 안내 메시지 출력
# ==========================================

class AutonomousUtils:
    def __init__(self, matrix_path='data/M.npy'):
        # 저장된 변환 행렬 로드
        try:
            self.M = np.load(matrix_path)
        except:
            print("경고: 변환 행렬 파일을 찾을 수 없습니다. Calibration을 먼저 진행하세요.")
            self.M = None

    def pixel_to_meter(self, px, py):
        """픽셀 좌표를 실제 미터 좌표로 변환"""
        if self.M is None:
            return 0, 0
            
        # 픽셀 좌표를 행렬 연산이 가능한 형태로 변환
        point = np.array([[[px, py]]], dtype=np.float32)
        # 투영 변환 적용
        transformed_point = cv2.perspectiveTransform(point, self.M)
        
        # 실제 x, y 미터 좌표 반환
        return transformed_point[0][0][0], transformed_point[0][0][1]

    def get_behavior_status(self, velocity):
        """속도에 따른 보행자 상태 및 박스 색상 결정"""
        abs_vel = abs(velocity)
        
        if abs_vel < VEL_STANDING:
            return "STANDING", (0, 0, 0)      # 검정색
        elif abs_vel < VEL_WALKING:
            return "WALKING", (0, 255, 255)   # 노란색
        else:
            return "RUNNING", (0, 0, 255)     # 빨간색

    def calculate_ttc(self, curr_y_m, velocity_y, cam_y_m):
        """
        curr_y_m: 보행자의 현재 y 위치 (meters)
        velocity_y: Y축 접근 속도 (양수면 다가옴, 음수면 멀어짐)
        """
        # 속도가 양수(+), 즉 나에게 다가오고 있을 때만 충돌 시간 계산
        if velocity_y > 0.01:
            # 내 위치(REAL_HEIGHT)에서 보행자 위치(curr_y_m)를 뺀 실제 남은 거리
            distance_left = cam_y_m - curr_y_m
            
            # 이미 내 위치를 지나쳤거나 겹쳤다면 0초 반환
            if distance_left <= 0:
                return 0.0
                
            ttc = distance_left / velocity_y
            return round(ttc, 1)
        
        # 뒤로 멀어지거나 제자리에 서 있으면 부딪힐 일 없음 (안전 값 반환)
        return 99.9

    def get_lane_guide(self, cx, frame_width):
        """차선 중심 이탈 여부 판단 및 안내 메시지 생성"""
        center_ratio = cx / frame_width
        
        if center_ratio < 0.5 - LANE_CENTER_OFFSET:
            return "STEER RIGHT >>", (0, 165, 255) # 오렌지색
        elif center_ratio > 0.5 + LANE_CENTER_OFFSET:
            return "<< STEER LEFT", (0, 165, 255)
        else:
            return "KEEP CENTER", (0, 255, 0)      # 초록색

    def get_lane_center(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(LANE_HSV_LOWER)
        upper = np.array(LANE_HSV_UPPER)
        
        # 이진 분류 (Mask 생성)
        mask = cv2.inRange(hsv, lower, upper)
        
        # 하단 영역(ROI)의 모멘트 계산하여 중심점 찾기
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            return cx, mask
        return None, mask

class Stabilizer:
    """Kalman Filter를 이용한 좌표 안정화 클래스"""
    def __init__(self):
        # 4개의 상태변수 (x, y, dx, dy) / 2개의 측정변수 (x, y)
        self.kf = cv2.KalmanFilter(4, 2) 
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32) 
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        
        # 노이즈 설정 (이 값을 조절해서 민감도를 바꿈)
        # Q가 작으면: 예측을 믿음 (부드러워짐, 느려짐)
        # Q가 크면: 센서를 믿음 (잘 튐, 빨라짐)
        self.kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03

    def update(self, x, y):
        # 1. 측정값 입력
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        
        # 2. 칼만 필터 계산 (Correct -> Predict)
        self.kf.correct(measured)
        predicted = self.kf.predict()
        
        # 3. 보정된 좌표 반환
        return predicted[0][0], predicted[1][0]