import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
# Initialize SIFT detector
sift = cv2.SIFT_create()

# Function to extract SIFT features
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# # Function to match features using kNN with OPENCV
# def match_features(descriptors1, descriptors2):
#     # Create BFMatcher object
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)  
#     # Apply ratio test
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append([m])
#     return good_matches

# Function to match features using kNN with sklearn
def match_features(descriptors1, descriptors2):
    # Initialize and train the kNN model
    knn = NearestNeighbors(n_neighbors=2).fit(descriptors2)

    # Find the 2 nearest neighbors
    distances, indices = knn.kneighbors(descriptors1)

    # Apply Lowe's ratio test
    '''
    Source: https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    Each keypoint of the first image is matched with a number of keypoints from the second image. We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement). Lowe's test checks that the two distances are sufficiently different. If they are not, then the keypoint is eliminated and will not be used for further calculations.
    '''
    good_matches = []
    for i, (d1, d2) in enumerate(distances):
        #if distance1 < distance2 * a_constant then keep this match
        if d1 < 0.75 * d2:
            # Create a DMatch object for each good match
            # match = cv2.DMatch(_queryIdx=i, _trainIdx=indices[i][0], _distance=d1)
            # good_matches.append([match])
            match = (i, indices[i][0], d1)
            good_matches.append(match)
    return good_matches

# Function to draw matches between two images for visualization for the SKLEARN method
def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    # Create a new output image that concatenates the two images together
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    output_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    output_image[0:h1, 0:w1] = img1
    output_image[0:h2, w1:] = img2

    # Draw lines between matches
    for match in matches:
        # match is a tuple: (index in keypoints1, index in keypoints2, distance)
        queryIdx, trainIdx, _ = match

        # Get the coordinates of the keypoints
        pt1 = tuple(map(int, keypoints1[queryIdx].pt))
        pt2 = tuple(map(int, keypoints2[trainIdx].pt))

        # Draw a line and circles for each match
        pt2 = (pt2[0] + w1, pt2[1])  # Adjust x-coordinate for pt2   
        #Needed to use openCV line function instead of matplotlib because matplotlib line function does not work with cv2 images
        cv2.line(output_image, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(output_image, pt1, 5, (255, 0, 0), -1)
        cv2.circle(output_image, pt2, 5, (255, 0, 0), -1)

    return output_image


# Load and process the fixed image
fixed_image_path = 'test_last_frame.png'  
fixed_image = cv2.imread(fixed_image_path)
fixed_keypoints, fixed_descriptors = extract_features(fixed_image)

# Open the video
video_path = 'trymefirst.mp4' 
cap = cv2.VideoCapture(video_path)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#save the video in the same directory as this file
# out = cv2.VideoWriter('feature_matched.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Extract features from the current frame
    frame_keypoints, frame_descriptors = extract_features(frame)
    # Match descriptors between fixed image and current frame
    matches = match_features(fixed_descriptors, frame_descriptors)
    print(f"Number of matches : {len(matches)}")
    # Draw matches
    # matched_frame = cv2.drawMatchesKnn(fixed_image, fixed_keypoints, frame, frame_keypoints, matches, None, flags=2, matchColor=(0,255,0))
    #Matched frame between fixed image and current frame
    matched_frame = draw_matches(fixed_image, fixed_keypoints, frame, frame_keypoints, matches)
    #To save the video uncomment the following line
    # out.write(matched_frame)

    # Display the frame
    cv2.imshow('Frame with SIFT Matches to Fixed Image', matched_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()
