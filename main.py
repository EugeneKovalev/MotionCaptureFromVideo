import imutils
import cv2
import numpy as np

kernel5 = np.array([[0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0]]).astype(np.uint8)  # 5x5 convolution kernel, round-ish

maxObjects = 100  # how many objects to detect track at once

procWidth = 136 # processing width (x resolution) of frame
fracF = 0.15  # adaptation fraction of background on each frame

noMotionCount = 0  # how many consecutive frames of no motion detected
motionCount = 0  # how many frames of consecutive motion
noMotionLimit = 5  # how many no-motion frames before start adapting

# run the webcam
camera = cv2.VideoCapture(0)
grabbed, frame = camera.read()  # get very first frame
if grabbed:
    frame = imutils.resize(frame, width=procWidth)  # resize to specified dimensions
    averageFrame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (15, 15), 0)

# loop over the frames of the video
# ===========================================================================

while grabbed:
    grabbed, frame = camera.read()

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=procWidth)
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (15, 15), 0)

    if noMotionCount > noMotionLimit:
        averageFrame = cv2.addWeighted(gray, fracF, averageFrame, (1.0 - fracF), 0)

    # difference of this frame from average background
    thresh = cv2.threshold(cv2.absdiff(averageFrame, gray), 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, kernel5, iterations=2)  # dilate to join adjacent regions, with larger kernel
    (_, cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motionDetect = False  # we have not yet found motion in this frame
    motionStart = False  # haven't just started to detect a motion
    i = -1  # count of detected objects

    # loop over the detected object-groups (contours)
    for c in cnts:
        xt, yt, wt, ht = cv2.boundingRect(c)
        i += 1  # found a large-enough object...
        if i >= maxObjects:  # ignore too-many objects
            continue

        # draw box around event
        cv2.rectangle(frame, (xt, yt), (xt + wt, yt + ht), (100, 100, 100), 2)

    noMotionCount = 0 if motionDetect else noMotionCount + 1
    motionCount = motionCount + 1 if motionDetect else 0

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

camera.release()
cv2.destroyAllWindows()
