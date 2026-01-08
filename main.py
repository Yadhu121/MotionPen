import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

prev_x, prev_y = None, None
canvas = None

hands = mp_hands.Hands (
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            lm = hand_landmarks.landmark[8]

            x = int(lm.x * w)
            y = int(lm.y * h)

            if canvas is None:
                canvas = frame.copy()
                canvas[:] = 0

            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)

            frame = cv2.add(canvas, frame)
            prev_x, prev_y = x, y

            cv2.circle(frame, (x,y), 10, (0, 0, 255), -1)

    else:
        prev_x, prev_y = None, None
        canvas = None

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()