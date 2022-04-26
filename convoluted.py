import cv2
import numpy as np

import detect


def edgy_af(img):
    """
    Apply crop, edge detection, and grayscale filters to input image
    """

    # Apply edge detection kernel
    h_kernel = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ]
    )
    v_kernel = np.array(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )
    img = cv2.filter2D(src=img, ddepth=-1, kernel=h_kernel) + cv2.filter2D(
        src=img, ddepth=-1, kernel=v_kernel
    )

    # Make grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


if __name__ == "__main__":
    image = cv2.imread("lfw/Abdul_Rahman/Abdul_Rahman_0001.jpg")
    img = edgy_af(image)

    print(img)
    # cv2.imshow("Original", image)
    # cv2.imshow("Kernel Edge", img)

    cv2.waitKey()
    cv2.imwrite("convoluted.jpg", img)
    cv2.destroyAllWindows()
