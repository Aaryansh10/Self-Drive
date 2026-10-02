import cv2 as cv
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "image.png")
image_path = os.path.normpath(image_path)

if not os.path.exists(image_path):
    print("Image path does not exist")

image = cv.imread(image_path)

def callback(x):
    pass

cv.namedWindow('Trackbars')
cv.resizeWindow('Trackbars', 640, 240)
cv.createTrackbar("Hue Min", "Trackbars", 0, 179, callback)
cv.createTrackbar("Hue Max", "Trackbars", 179, 179, callback)
cv.createTrackbar("Sat Min", "Trackbars", 0, 255, callback)
cv.createTrackbar("Sat Max", "Trackbars", 255, 255, callback)
cv.createTrackbar("Val Min", "Trackbars", 0, 255, callback)
cv.createTrackbar("Val Max", "Trackbars", 255, 255, callback)

while True:
    blurred_image = cv.GaussianBlur(image, (5, 5), 0)
    hsv = cv.cvtColor(blurred_image, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv.getTrackbarPos("Val Max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv.inRange(hsv, lower, upper)
    
    result = cv.bitwise_and(image, image, mask=mask)
    cv.imshow("Original", image)
    cv.imshow("Mask", mask)
    cv.imshow("Result", result)

    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

cv.destroyAllWindows()