import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
hands  = mp_hands.Hands(min_tracking_confidence=0.3)

while True:
    sucess,frame=cam.read()
    result = hands.process(frame)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_drawings.draw_landmarks(image=frame,landmark_list=hand_landmark,connections=mp_hands.HAND_CONNECTIONS)


    cv2.imshow('camera',frame)
    if cv2.waitKey(1) & 0XFF==27:
        break
cam.release
cv2.destroyAllWindows()
