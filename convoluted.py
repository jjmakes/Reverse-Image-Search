import cv2
import numpy as np

image = cv2.imread("lfw/Abdul_Rahman/Abdul_Rahman_0001.jpg")

# Print error message if image is null
if image is None:
    print("Could not read image")

# Apply identity kernel
kernel1 = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
)

identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

cv2.imshow("Original", image)
cv2.imshow("Identity", identity)

cv2.waitKey()
cv2.imwrite("identity.jpg", identity)
cv2.destroyAllWindows()

# Apply edge detection kernel
kernel2 = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]
)
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

cv2.imshow("Original", image)
cv2.imshow("Kernel Edge", img)

cv2.waitKey()
cv2.imwrite("convoluted.jpg", img)
cv2.destroyAllWindows()
