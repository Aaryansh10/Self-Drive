import os
import sys
import numpy as np
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))

workspace_src = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(workspace_src)

from vision_py.vision_py import hsv_masking

image_path = os.path.join(script_dir, "image.png")
image_path = os.path.normpath(image_path)

if not os.path.exists(image_path):
    print("Image path does not exist")

image = cv2.imread(image_path)

mask = hsv_masking.generate_mask(image)
mask_temp = mask[0] | mask[1] | mask[2]
result = cv2.bitwise_and(image, image, mask=mask_temp)
mask = np.transpose(mask, (1, 2, 0))

cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Result", result)

key = cv2.waitKey(0) & 0xFF

cv2.destroyAllWindows()