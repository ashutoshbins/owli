import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture classification logic
GESTURES = {
    "Fist": lambda lm: all(lm[i][1] > lm[i + 1][1] for i in range(5, 20, 4)),
    "Palm": lambda lm: all(lm[i][1] < lm[i + 1][1] for i in range(5, 20, 4)),
    "Thumbs Up": lambda lm: lm[4][0] < lm[3][0] and all(lm[i][1] > lm[i + 1][1] for i in range(5, 17, 4)),
}

def recognize_gesture(landmarks):
    lm = [(lm.x, lm.y) for lm in landmarks.landmark]
    for gesture, check in GESTURES.items():
        if check(lm):
            return gesture
    return "Unknown"

def process_frame(frame):
    # Convert frame to RGB
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe Hands
    result = hands.process(rgb_img)
    gesture_detected = "None"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_detected = recognize_gesture(hand_landmarks)
            cv2.putText(img, gesture_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Live Gesture Recognition on Streamlit Cloud")

webrtc_streamer(
    key="gesture-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    force_relay=True  # ✅ Fixes WebRTC connection issues
)



