import numpy as np
import cv2

WHITE_LOWER = np.array([0, 0, 235])
WHITE_UPPER = np.array([179, 16, 255])
YELLOW_LOWER = np.array([24, 45, 0])
YELLOW_UPPER = np.array([45, 255, 255])
MIN_AREA = 50
MAX_CIRCULARITY_WHITE = 0.3
MAX_CIRCULARITY_YELLOW = 0.7
MIN_ASPECT_RATIO = 5.0

def clean_mask(mask, kernel, is_white):
    mask_clean = np.zeros_like(mask, dtype=np.uint8) 

    open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = ((4 * np.pi * area) / (perimeter * perimeter))
        if circularity >= MAX_CIRCULARITY_WHITE and is_white:
            continue

        if circularity >= MAX_CIRCULARITY_YELLOW and not is_white:
            continue
        
        cv2.drawContours(mask_clean, [contour], 0, 1, -1)

    return mask_clean

def extract_stopline(mask):
    if mask is None:
        return

    mask_stopline = np.zeros_like(mask, dtype=np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y , w, h = cv2.boundingRect(contour)
        
        aspect_ratio = w / float(h)
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        cv2.drawContours(mask_stopline, [contour], 0, 1, -1)

    return mask_stopline

def generate_mask(img):
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    mask_white = cv2.inRange(hsv_img, WHITE_LOWER, WHITE_UPPER)
    mask_yellow = cv2.inRange(hsv_img, YELLOW_LOWER, YELLOW_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_white = clean_mask(mask_white, kernel, True)
    mask_yellow = clean_mask(mask_yellow, kernel, False)

    mask_stopline = extract_stopline(mask_white)
    mask_whitelane = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_stopline))

    mask_final = np.stack((mask_whitelane, mask_yellow, mask_stopline), axis=0)

    return mask_final