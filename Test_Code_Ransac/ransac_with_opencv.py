import cv2
import numpy as np

n_keypoints = 200
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(
                nfeatures=n_keypoints,  # Set to 0 to disable limiting the number of keypoints
                # nOctaveLayers=3, # Number of layers in each octave of the image. Default: 3. (Automatically determined by image size)
                contrastThreshold=0.04, # If contrast of keypoint below this threshold, won't be detected. Default: 0.04
                edgeThreshold=10, #  If difference in intensity between keypoint and surrounding pixels below threshold, keypoint rejected. Default: 10
                sigma=1.6 # S.D. of Gaussian on first octave. Default: 1.6
            )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def find_homography(keypoints1, keypoints2, matches):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

fixed_image_path = 'test_last_frame.png'
fixed_image = cv2.imread(fixed_image_path)
fixed_keypoints, fixed_descriptors = extract_features(fixed_image)

video_path = 'trymefirst.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_keypoints, frame_descriptors = extract_features(frame)
    matches = match_features(fixed_descriptors, frame_descriptors)

    if len(matches) > 4:  # Homography needs at least 4 matches
        H, mask = find_homography(fixed_keypoints, frame_keypoints, matches)
        # You can use H for further processing, like warping images
        # Uncomment below to draw matches used for homography
        # matches = [matches[i] for i in range(len(matches)) if mask[i]]

    matched_frame = cv2.drawMatches(fixed_image, fixed_keypoints, frame, frame_keypoints, matches, None, flags=2)

    cv2.imshow('Frame with SIFT Matches to Fixed Image', matched_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
