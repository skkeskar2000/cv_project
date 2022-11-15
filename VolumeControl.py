import cv2
import time
import numpy as np
import HandTrackingModule as htm
from cvzone.HandTrackingModule import HandDetector
import cvzone
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

pTime = 0

detector = htm.handDetector(detectionCon=0.8,maxHands=2)
detector2 = HandDetector(detectionCon=0.8,maxHands=2)

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
coff = np.polyfit(x, y, 2) 

while True:
    success, img = cap.read()
    hands = detector2.findHands(img, draw=False)
    img = detector.findHands(img)

    closehand = []
    if hands:

        for i in range(len(hands)):
            lmList = hands[i]['lmList']
            x, y, w, h = hands[i]['bbox']
            x1, y1, z1 = lmList[5]
            x2, y2, z2 = lmList[17]

            distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            A, B, C = coff
            distanceCM = A * distance ** 2 + B * distance + C
            closehand.append(distanceCM)


    x = len(closehand)
    if (x == 0):
        position = detector.findPosition(img, draw=False)
        print("No hands detected")
    elif (x == 1):
        position = detector.findPosition(img, draw=False)
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (50, 100))
    else:
        if (closehand[0] > closehand[1]):
            position = detector.findPosition(img,1, draw=False)
            cvzone.putTextRect(img, f'{int(closehand[1])} cm', (50, 100))
        else:
            position = detector.findPosition(img, draw=False)
            cvzone.putTextRect(img, f'{int(closehand[0])} cm', (50, 100))
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

        vol = np.interp(length, [50, 235], [minVol, maxVol])

        volBar = np.interp(length, [50, 235], [400, 150])
        volPer = np.interp(length, [50, 235], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    key =cv2.waitKey(1)
    key
    if key == 27:
        print("Esc is pressed")
        break