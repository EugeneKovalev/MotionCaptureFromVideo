import imutils
import cv2
import numpy as np

# neccesary convolution matrix
kernel = np.array([[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]]).astype(np.uint8)
# print("Enter the number of max objects:")
maxObjects = 1 #input()
frameWidth = 400
motionlessFramesCount = 0
motionFramesCount = 0
motionlessFramesLimit = 5

# run the webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BRIGHTNESS, 100)
camera.set(cv2.CAP_PROP_SATURATION, 100)
camera.set(cv2.CAP_PROP_CONTRAST, 10)
grabbed, frame = camera.read()
if grabbed:
    frame = imutils.resize(frame, width=frameWidth)  # resize to specified dimensions
    averageFrame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (15, 15), 0)
while grabbed:
    grabbed, frame = camera.read()
    frame = imutils.resize(frame, width=frameWidth)
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (15, 15), 0)

    if motionlessFramesCount > motionlessFramesLimit:
        averageFrame = cv2.addWeighted(gray, 0.15, averageFrame, (1.0 - 0.15), 0)

    # diff of this frame from avg background
    thresh_hold = cv2.threshold(cv2.absdiff(averageFrame, gray), 25, 255, cv2.THRESH_BINARY)[1]
    thresh_hold = cv2.dilate(thresh_hold, kernel, iterations=2)
    _, contours, hierarchy = cv2.findContours(thresh_hold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motionDetected = False
    motionStarted = False
    detectedObjectsCount = -1

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        detectedObjectsCount += 1
        if detectedObjectsCount >= maxObjects:
            continue
        # draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)

    motionlessFramesCount = 0 if motionDetected else motionlessFramesCount + 1
    motionFramesCount = motionFramesCount + 1 if motionDetected else 0

    cv2.imshow("MyVideo", frame)
    key = cv2.waitKey(1) & 0xFFF

camera.release()
cv2.destroyAllWindows()
