import cv2
import mediapipe as mp
import pyautogui
import time

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 중복 입력 방지를 위한 변수
last_action_time = 0
cooldown = 2  # 2초 동안 재입력 방지
v_start_time = 0        # V 제스처가 시작된 시각
required_duration = 0.8 # 최소 유지 시간 (1초)
gesture_active = False  # 현재 V 제스처가 진행 중인지 여부

def is_v_sign(hand_landmarks):
    # 손가락 끝(Tip) 번호: 검지(8), 중지(12), 약지(16), 새끼(20)
    # 각 손가락의 끝이 바로 아래 마디보다 위에 있는지 확인
    finger_8_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    finger_12_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    finger_16_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    finger_20_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    
    # 검지와 중지만 펴져있으면 V 사인으로 간주
    return finger_8_up and finger_12_up and finger_16_down and finger_20_down

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1) # 좌우 반전
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    v_count = 0
    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_v_sign(hand_landmarks):
                v_count += 1

    if v_count > 0:
            if not gesture_active:
                # 방금 막 V를 시작함
                v_start_time = current_time
                gesture_active = True
            
            # 얼마나 유지되었는지 계산
            elapsed = current_time - v_start_time
            
            # 쿨다운이 끝났고, 설정한 시간(1초) 이상 유지했다면 실행
            if current_time - last_action_time > cooldown:
                if elapsed >= required_duration:
                    if v_count == 2:
                        print("양손 V 1초 유지: 이전 페이지")
                        pyautogui.press('left')
                    elif v_count == 1:
                        print("한 손 V 1초 유지: 다음 페이지")
                        pyautogui.press('right')
                    
                    last_action_time = current_time # 쿨다운 시작
                    gesture_active = False          # 실행 후 초기화
    else:
        # V 사인이 화면에서 사라지면 타이머 리셋
        gesture_active = False
        v_start_time = 0

    # 시각적 피드백 (화면에 게이지나 상태 표시)
    if gesture_active and (current_time - last_action_time > cooldown):
        progress = min((current_time - v_start_time) / required_duration, 1.0)
        cv2.rectangle(image, (10, 70), (210, 90), (255, 255, 255), 2)
        cv2.rectangle(image, (10, 70), (10 + int(progress * 200), 90), (0, 255, 0), -1)
        cv2.putText(image, "Holding...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(image, f"V Count: {v_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Improved Gesture Control', image)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()