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

        # 제스처 판별 로직
        if current_time - last_action_time > cooldown:
            if v_count == 2:
                print("양손 V 감지: 이전 페이지로 (Left)")
                pyautogui.press('left')
                last_action_time = current_time
            elif v_count == 1:
                print("한 손 V 감지: 다음 페이지로 (Right)")
                pyautogui.press('right')
                last_action_time = current_time

    # 화면에 상태 표시
    cv2.putText(image, f"V Count: {v_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('PPT Gesture Control', image)

    if cv2.waitKey(1) & 0xFF == 27: # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()