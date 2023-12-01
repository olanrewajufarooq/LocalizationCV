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

# Load and process the fixed image
fixed_image_path = 'test.jpg'  
fixed_image = cv2.imread(fixed_image_path)
fixed_keypoints, fixed_descriptors = extract_features(fixed_image)

# Open the video
video_path = 'undistorted_video.avi' 
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('feature_matched.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_keypoints, frame_descriptors = extract_features(frame)
    # Match descriptors between fixed image and current frame
    matches = match_features(fixed_descriptors, frame_descriptors)

    # Draw matches
    matched_frame = cv2.drawMatchesKnn(fixed_image, fixed_keypoints, frame, frame_keypoints, matches, None, flags=2, matchColor=(0,255,0))
    out.write(matched_frame)
    # Display the frame
    cv2.imshow('Frame with SIFT Matches to Fixed Image', matched_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()
