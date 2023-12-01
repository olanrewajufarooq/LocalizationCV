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

def calculate_homography(points_array1, points_array2):
    # Validate input
    if len(points_array1) != len(points_array2):
        raise ValueError("Point arrays must have the same number of points.")

    # Build the matrix A for the homography calculation
    A = []

    for point1,point2 in zip(points_array1,points_array2):
        x1, y1 = point1
        x2, y2 = point2
        A.append([x1 , y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A,dtype=np.double)
    eig_values , eig_vectors = np.linalg.eig(A.T @ A)

    min_eig_value = np.argmin(eig_values)
    min_eig_value_index = np.unravel_index(min_eig_value, eig_values.shape)
    H_matrix = eig_vectors[:,min_eig_value_index].reshape(3,3)

    H_matrix = H_matrix / H_matrix[2,2]

    return H_matrix

def manual_warpPerspective(src_img, H_matrix, output_shape):
    # Calculate the inverse of the transformation matrix
    inv_trans_matrix = np.linalg.inv(H_matrix)

    # Determine the number of channels in the source image
    num_channels = 1 if len(src_img.shape) == 2 else src_img.shape[2]

    # Create an empty output image
    if num_channels == 1:
        dst_img = np.zeros((output_shape[0], output_shape[1]), dtype=src_img.dtype)
    else:
        dst_img = np.zeros((output_shape[0], output_shape[1], num_channels), dtype=src_img.dtype)

    # Iterate over each pixel in the output image
    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            # Apply the inverse transformation matrix to get the corresponding source pixel
            src_x, src_y, w = inv_trans_matrix @ np.array([x, y, 1])
            src_x = int(src_x / w)
            src_y = int(src_y / w)

            # Check if the source pixel is within the bounds of the source image
            if 0 <= src_x < src_img.shape[1] and 0 <= src_y < src_img.shape[0]:
                if num_channels == 1:
                    dst_img[y, x] = src_img[src_y, src_x]
                else:
                    dst_img[y, x, :] = src_img[src_y, src_x, :]

    return dst_img

#Load the images
image1_path = 'parede1.jpg'
image2_path = 'parede2.jpg'
points_array1 = get_points_from_image(image1_path)
points_array2 = get_points_from_image(image2_path)

# Convert the lists to NumPy arrays
points_array1 = np.array(points_array1, dtype=np.double)
points_array2 = np.array(points_array2, dtype=np.double)

# Find the homography matrix
H_matrix = calculate_homography(points_array1, points_array2)


# Load the images
image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
image1 = np.array(image1)
image2 = np.array(image2)

#height and width of images
'''
took help from https://github.com/ZohebAbai/mobile_sensing_robotics/blob/main/Visual_Features_RANSAC.ipynb
to stitch the images
'''
width = image1.shape[1] + image2.shape[1]
height = image1.shape[0] + image2.shape[0]
# Warp image 1 to image 2
result12 = manual_warpPerspective(image2, H_matrix, (height,width))
result12[0:image1.shape[0], 0:image1.shape[1]] = image1
# Warp image 2 to image 1
result21 = manual_warpPerspective(image1, H_matrix, (height,width))
result21[0:image2.shape[0], 0:image2.shape[1]] = image2
image_togehter = np.concatenate((result12, result21),axis=1)

# Display both images
cv2.imshow("Homography", image_togehter)
cv2.waitKey(0)
cv2.destroyAllWindows()
