import cv2 as cv
import numpy as np

# Load the input image
image = cv.imread("campus_shadow.jpeg")
image = cv.resize(image, (1500, 1000))

# Convert the image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, threshold = cv.threshold(gray, 100, 400, cv.THRESH_BINARY)

# Create a mask by inverse thresholding
image = cv.bitwise_not(threshold)

mask = np.zeros_like(image)
v = np.array([[693, 319], [314, 995], [1493, 996], [1495, 693], [894, 327]])
cv.fillPoly(mask, [v], (255, 255, 255))
image = cv.bitwise_and(image, mask)

cv.imwrite("output_image.jpeg", image)

rgb_image = cv.imread('campus_shadow.jpeg', 1)
rgb_image = cv.resize(rgb_image, (1500, 1000))

mask = cv.imread('output_image.jpeg', 0)

final = cv.inpaint(rgb_image, mask, 250, cv.INPAINT_TELEA)
cv.imwrite("shadow_removal1.jpeg", final)