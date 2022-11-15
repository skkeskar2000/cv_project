import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time
from cvzone.HandTrackingModule import HandDetector
import cvzone
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandTrackingModule as htm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return cv2.flip(image, 1)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:

    detector2 = HandDetector(detectionCon=0.8, maxHands=2)
    detector = htm.handDetector(detectionCon=0.8,maxHands=2)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()

    volRange = volume.GetVolumeRange()

    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPer = 0
    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
    pTime = 0
    cTime = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = process(img)
        
        # FPS
        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Hand Distance

        hands = self.detector2.findHands(img, draw=False)
        image = self.detector.findHands(img)

        closehand = []
        if hands:
            for i in range(len(hands)):

                lmList = hands[i]['lmList']
                x, y, w, h = hands[i]['bbox']
                x1, y1 , z1 = lmList[5]
                x2, y2 , z2 = lmList[17]

                distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
                A, B, C = self.coff
                distanceCM = A * distance ** 2 + B * distance + C
                closehand.append(distanceCM)
                print(closehand)

        # print(distanceCM, distance)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)

        numOfHands = len(closehand)
        if(numOfHands==0):
            position = self.detector.findPosition(image, draw=False)
            print("No hands detected")
        elif(numOfHands==1):
            position = self.detector.findPosition(image, draw=False)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (50,100))
        else:
            if(closehand[0]>closehand[1]):
                position = self.detector.findPosition(img, 1,draw=False)
                cvzone.putTextRect(img, f'{int(closehand[1])} cm', (50,100))
            else:
                position = self.detector.findPosition(img, draw=False)
                cvzone.putTextRect(img, f'{int(closehand[0])} cm', (50,100))
       

        if len(position) != 0:

            xa1, ya1 = position[4][1], position[4][2]
            xa2, ya2 = position[8][1], position[8][2]
            cx, cy = (xa1 + xa2) // 2, (ya1 + ya2) // 2


            cv2.circle(img, (xa1, ya1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (xa2, ya2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (xa1, ya1), (xa2, ya2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(xa2 - xa1, ya2 - ya1)

        # Hand range 50 - 235
        # Volume Range -65 - 0

            self.vol = np.interp(length, [50, 235], [self.minVol, self.maxVol])

            self.volBar = np.interp(length, [50, 235], [400, 150])
            self.volPer = np.interp(length, [50, 235], [0, 100])

            self.volume.SetMasterVolumeLevel(self.vol, None)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(self.volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(self.volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)


        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
