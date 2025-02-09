import streamlit as st
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def recognize_gesture(landmarks):
    lm = [(lm.x, lm.y) for lm in landmarks.landmark]
    for gesture, check in GESTURES.items():
        if check(lm):
            return gesture
    return "Unknown"

def main():
    st.title("Live Gesture Recognition using MediaPipe")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        gesture_detected = "None"
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_detected = recognize_gesture(hand_landmarks)
                cv2.putText(frame, gesture_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        FRAME_WINDOW.image(frame, channels="BGR")
        st.write("Detected Gesture:", gesture_detected)

if __name__ == "__main__":
    main()
