import cv2 as cv
import numpy as np
import  asyncio
import threading
import time

cap = cv.VideoCapture(cv.CAP_V4L)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("x",gray)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

