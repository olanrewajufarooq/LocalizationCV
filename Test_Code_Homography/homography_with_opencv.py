import cv2
import numpy as np
from PIL import Image
def get_points_from_image(image_path):
    # Function to handle mouse clicks
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            points.append((x, y))
            cv2.imshow("image", img)

    # Load the image
    img = cv2.imread(image_path)
    cv2.imshow("image", img)

    # List to store points
    points = []

    # Set mouse callback function
    cv2.setMouseCallback("image", click_event)

    # Wait until any key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points

#Load the images
image1_path = 'test_frame1.png'
image2_path = 'test_last_frame.png'
points_array1 = get_points_from_image(image1_path)
points_array2 = get_points_from_image(image2_path)

# Convert the lists to NumPy arrays
points_array1 = np.array(points_array1, dtype=np.double)
points_array2 = np.array(points_array2, dtype=np.double)

# Find the homography matrix
h, status = cv2.findHomography(points_array1, points_array2, cv2.RANSAC, 5.0)


# Load the images
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

image1 = np.array(image1)
image2 = np.array(image2)

#height and width of images
width = image1.shape[1] + image2.shape[1]
height = image1.shape[0] + image2.shape[0]
# Warp image 1 to image 2
result = cv2.warpPerspective(np.array(image1), h, (width, height))

# Display both images
result[0:image2.shape[0], 0:image2.shape[1]] = image2
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

cv2.imshow("Warped Source Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
