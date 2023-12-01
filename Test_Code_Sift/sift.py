import cv2
import os
import glob
#initiate sift detector
sift = cv2.SIFT.create()

#Open the video
image_folder = '/Users/mdsazidurrahman/Computer Vision/Test features/project/Tesla/TeslaVC_carreira/undistorted_images/2023-07-23_11-36-50-back'
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
print(image_paths)

video_path = 'output_video.avi'
frame_rate = 2
first_frame = cv2.imread(image_paths[0])
height, width, layers = first_frame.shape
video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))



# for img_path in image_paths:
#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Detect SIFT features
#     keypoints, _ = sift.detectAndCompute(gray, None)

#     # Draw keypoints for visualization
#     frame_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

#     # Write frame to video
#     video.write(frame_with_keypoints)
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

keypoints, desc = sift.detectAndCompute(gray, None)
frame_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0),)
print(desc.shape)
print(len(keypoints))

output_path = 'img_withkeypoint.jpg'  # Replace with the path where you want to save the image
cv2.imwrite(output_path, frame_with_keypoints)
cv2.imshow('frame', frame_with_keypoints)

cv2.waitKey(0)

# video.release()
cv2.destroyAllWindows()