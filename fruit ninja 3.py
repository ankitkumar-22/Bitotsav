import cv2
import mediapipe as mp
import pyautogui
import mouse


def print_message(frame, thumb, left):
    if thumb == True:
        message = "activated"
    else:
        message = "deactivated"

    if left == True:
        hand = "left"
    else:
        hand = "right"
    cv2.putText(frame, f"{message}   {hand}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def detect_hands(img):
    hand_detector = mp.solutions.hands.Hands()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_img)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            mp.solutions.drawing_utils.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                # id contains the index of landmark
                # landmark contains x,y,z co ordinate for that landmark
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:
                    cv2.circle(frame, (x, y), 10, (0, 0, 255))
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    pyautogui.moveTo(index_x, index_y)

                # conditon to check the hand
                if hand.landmark[5].x > hand.landmark[
                    17].x:  # Check the x-coordinates of landmarks 5 (index) and 17 (pinky)
                    is_left_hand = True
                else:
                    is_left_hand = False
                # conditon to check if thumb is open or not
                is_thumb_open = False
                thumb_base = hand.landmark[2]  # Index 2 represents the base of the thumb
                if is_left_hand:
                    if hand.landmark[4].x > thumb_base.x:
                        is_thumb_open = True
                else:
                    if hand.landmark[4].x < thumb_base.x:
                        is_thumb_open = True

                print_message(img, is_thumb_open, is_left_hand)
                if is_thumb_open:
                    pyautogui.mouseDown()



cap = cv2.VideoCapture(0)
frame_width, frame_height = 640, 480
screen_width, screen_height = pyautogui.size()
cap.set(3, frame_width)
cap.set(4, frame_height)
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    detect_hands(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
