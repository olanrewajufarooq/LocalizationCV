import cv2
import numpy as np

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Function to extract SIFT features
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Function to match features using kNN
def match_features(descriptors1, descriptors2):
    # Create BFMatcher object
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    return good_matches

# Open the video
video_path = 'undistorted_video.avi'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, prev_frame = cap.read()
prev_keypoints, prev_descriptors = extract_features(prev_frame)

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_keypoints, curr_descriptors = extract_features(curr_frame)

    # Match descriptors
    matches = match_features(prev_descriptors, curr_descriptors)

    # Draw matches
    matched_frame = cv2.drawMatchesKnn(prev_frame, prev_keypoints, curr_frame, curr_keypoints, matches, None, flags=2)

    # Display the frame
    cv2.imshow('Frame with SIFT Matches', matched_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update previous frame and descriptors
    prev_frame = curr_frame
    prev_keypoints = curr_keypoints
    prev_descriptors = curr_descriptors

# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()
