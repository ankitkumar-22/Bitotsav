import mediapipe as mp
import cv2
import keyboard
import time

def calculate_bounding_box(frame,hand_landmarks):
    h,w = frame.shape[:2]
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    # Calculate bounding box coordinates
    xmin = int(min(x_coords))
    xmax = int(max(x_coords))
    ymin = int(min(y_coords))
    ymax = int(max(y_coords))
    return ((xmin, ymin), (xmax, ymax))


def count_fingers(hand_landmarks):
    # Define landmarks for each finger
    finger_landmarks = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]
    num_fingers = 0

    # Check if hand is left or right based on landmark positions
    if hand_landmarks.landmark[5].x > hand_landmarks.landmark[17].x:  # Check the x-coordinates of landmarks 5 (index) and 17 (pinky)
        is_left_hand = True
    else:
        is_left_hand = False

    is_thumb_open = False
    # Get the landmark for the base of the thumb
    thumb_base = hand_landmarks.landmark[2]  # Index 2 represents the base of the thumb
    if is_left_hand:
        if hand_landmarks.landmark[4].x > thumb_base.x:
            is_thumb_open = True
    else:
        if hand_landmarks.landmark[4].x < thumb_base.x:
            is_thumb_open = True


    if is_thumb_open:
        num_fingers += 1

    # Count the number of fingers shown by the hand

    for finger_idx in finger_landmarks:
        is_finger_up = True
        # For other fingers, check y-coordinate of finger tip
        for landmark_idx in finger_idx[1:]:
            if hand_landmarks.landmark[landmark_idx].y < hand_landmarks.landmark[finger_idx[0]].y:
                is_finger_up = False
                break
        if is_finger_up:
            num_fingers += 1
    return num_fingers

import threading

def key_release(key):
    keyboard.release(key)

def key_press(num_fingers, hand_counter):
    if hand_counter == 1:
        if num_fingers == 1:
            keyboard.press('d')
            threading.Timer(.5, key_release, args=('d',)).start()
        if num_fingers == 2:
            keyboard.press('w')
            threading.Timer(.5, key_release, args=('w',)).start()
        if num_fingers == 3:
            keyboard.press('a')
            threading.Timer(.5, key_release, args=('a',)).start()

    if hand_counter == 2:
        if num_fingers == 1:
            keyboard.press('h')
            threading.Timer(.5, key_release, args=('h',)).start()
        if num_fingers == 2:
            keyboard.press('t')
            threading.Timer(.5, key_release, args=('t',)).start()
        if num_fingers == 3:
            keyboard.press('f')
            threading.Timer(.5, key_release, args=('f',)).start()



'''
def detect_hand_gestures(frame,str):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    mp_drawing = mp.solutions.drawing_utils
    #hand_counter = 1
    if str=="left":
        hand_counter= 1
    else:
        hand_counter = 2
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            bbox = calculate_bounding_box(frame, hand_landmarks)
            cv2.rectangle(frame, bbox[0], bbox[1],(0,255,0),3)
            text_position = (bbox[0][0], bbox[0][1] - 10)
            num_fingers = count_fingers(hand_landmarks)
            key_press(num_fingers,hand_counter)
            cv2.putText(frame, f"Player {hand_counter} , fingers {num_fingers}",text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_counter += 1
'''
def detect_hand_gestures(frame, hand_counter):
    # Instantiate Hands object outside the loop to avoid repeated instantiation
    hands = mp.solutions.hands.Hands()

    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    results = hands.process(rgb_frame)

    # Import drawing utils outside the loop
    mp_drawing = mp.solutions.drawing_utils

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate bounding box
            bbox = calculate_bounding_box(frame, hand_landmarks)

            # Draw bounding box
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 3)

            # Define text position for hand information
            text_position = (bbox[0][0], bbox[0][1] - 10)

            # Count fingers
            num_fingers = count_fingers(hand_landmarks)

            # Perform actions based on finger count
            key_press(num_fingers, hand_counter)

            # Display hand information
            cv2.putText(frame, f"Player {hand_counter}, fingers {num_fingers}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Increment hand counter
            hand_counter += 1

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
desired_fps = 5
prev_time = time.time()
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    half_width = frame.shape[1] // 2
    height = frame.shape[0]
    left_half = frame[:, :half_width]
    right_half = frame[:, half_width:]
    detect_hand_gestures(left_half,1)
    detect_hand_gestures(right_half,2)
    cv2.line(frame, [half_width,0], [half_width,height] ,(0,255,255) ,5)
    cv2.imshow("Frame", frame)
    curr_time = time.time()
    elapsed_time = curr_time - prev_time
    time_to_wait = 1.0 / desired_fps - elapsed_time
    if time_to_wait > 0:
        time.sleep(time_to_wait)
    # Update previous time
    prev_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
