import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

prev_x, prev_y = None, None
smooth_x, smooth_y = None, None
alpha = 0.7
canvas = None

hands = mp_hands.Hands (
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.8
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

            if canvas is None:
                canvas = frame.copy()
                canvas[:] = 0
            
            h, w, _ = frame.shape

            index = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]
            middle = hand_landmarks.landmark[12]
            pinky = hand_landmarks.landmark[20]

            ix = int(index.x * w)
            iy = int(index.y * h)
            tx = int(thumb.x * w)
            ty = int(thumb.y * h)
            mx = int(middle.x * w)
            my = int(middle.y * h)
            px = int(pinky.x * w)
            py = int(pinky.y * h)

            dist = ((ix - tx)**2 + (iy - ty)**2) ** 0.5
            TOUCH_DIST = 30
            dist_erase = ((tx - mx)**2 + (ty - my)**2) ** 0.5
            ERASE_DIST = 30
            dist_clear = ((tx - px)**2 + (ty - py)**2) ** 0.5
            CLEAR_DIST = 30

            if smooth_x is None:
                smooth_x, smooth_y = ix, iy
            else:
                smooth_x = int(alpha * smooth_x + (1 - alpha) * ix)
                smooth_y = int(alpha * smooth_y + (1 - alpha) * iy)

            if dist < TOUCH_DIST:
                cv2.circle(frame, (smooth_x,smooth_y), 10, (0, 255, 0), -1)
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), (0, 0, 255), 8)

                frame = cv2.add(canvas, frame)
                prev_x, prev_y = smooth_x, smooth_y

            elif dist_erase < ERASE_DIST:
                cv2.circle(frame, (smooth_x, smooth_y), 5, (255,0,0), -1)
                cv2.circle(canvas, (smooth_x, smooth_y), 15, (0,0,0), -1)
                prev_x, prev_y = None, None

            elif dist_clear < CLEAR_DIST:
                cv2.circle(frame, (smooth_x, smooth_y), 5, (255,255,255), -1)
                canvas[:] = 0
                prev_x, prev_y = None, None

            else:
                cv2.circle(frame, (smooth_x, smooth_y), 5, (0, 0, 255), -1)
                prev_x, prev_y = None, None

    else:
        prev_x, prev_y = None, None
    
    if canvas is not None:
        frame = cv2.add(canvas, frame)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()