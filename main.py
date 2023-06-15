# Import necessary libraries
from cv2 import cv2 as cv
import numpy as np


# A function to increse brightness of given image
def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


# A function to draw rectangle
def drawRec(biggestNew, mainFrame):
    cv.line(mainFrame, (biggestNew[0][0][0], biggestNew[0][0][1]), (biggestNew[1][0][0], biggestNew[1][0][1]),
            green, 20)
    cv.line(mainFrame, (biggestNew[0][0][0], biggestNew[0][0][1]), (biggestNew[2][0][0], biggestNew[2][0][1]),
            green, 20)
    cv.line(mainFrame, (biggestNew[3][0][0], biggestNew[3][0][1]), (biggestNew[2][0][0], biggestNew[2][0][1]),
            green, 20)
    cv.line(mainFrame, (biggestNew[3][0][0], biggestNew[3][0][1]), (biggestNew[1][0][0], biggestNew[1][0][1]),
            green, 20)


green = (0, 255, 0)
purple = (255, 0, 255)
w, h = 480, 640

# write image folder name
img = cv.imread("image.jpeg")
warp = img.copy()
img = cv.resize(img, (int(w * 2), int(h * 2)))
img_bright = increase_brightness(img, value=70)
gray = cv.cvtColor(img_bright, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3, 3), 3)

ret, thresh1 = cv.threshold(blur, 140, 255, cv.THRESH_BINARY)

contours, _ = cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
thresh1 = cv.resize(thresh1, (w, h))

contour = img.copy()
contour = cv.drawContours(contour, contours, -1, purple, 4)
corner = img.copy()
maxArea = 0
biggest = []
# Selecting biggest rectangle
for i in contours:
    area = cv.contourArea(i)
    if area > 1000:
        peri = cv.arcLength(i, True)
        edges = cv.approxPolyDP(i, 0.04 * peri, True)
        if area > maxArea and len(edges) == 4:
            biggest = edges
            maxArea = area

# Perspective transform
if len(biggest) != 0:
    biggest = biggest.reshape((4, 2))
    biggestNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = biggest.sum(1)
    biggestNew[0] = biggest[np.argmin(add)]
    biggestNew[3] = biggest[np.argmax(add)]
    diff = np.diff(biggest, axis=1)
    biggestNew[1] = biggest[np.argmin(diff)]
    biggestNew[2] = biggest[np.argmax(diff)]
    drawRec(biggestNew, corner)
    corner = cv.drawContours(corner, biggestNew, -1, purple, 8)
    point1 = np.float32(biggestNew)
    point2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv.getPerspectiveTransform(point1, point2)
    warp = cv.warpPerspective(img, matrix, (w, h))

img = cv.resize(img, (w, h))
contour = cv.resize(contour, (w, h))
corner = cv.resize(corner, (w, h))

cv.imshow("img", img)
cv.imshow("contour", contour)
cv.imshow("corner", corner)
cv.imshow("warp", warp)
cv.imwrite("warp.jpeg", warp)
cv.waitKey(0)
