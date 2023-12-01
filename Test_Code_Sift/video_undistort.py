import cv2
from scipy.io import loadmat
import numpy as np
# Load the video
video_path = '/Users/mdsazidurrahman/Computer Vision/Test features/project/Tesla/TeslaVC_carreiraVIDEOS/2023-07-23_11-36-50-back.mp4'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('undistorted_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Camera calibration parameters
camera_matrix = np.array([
    [519.4039,   0,      656.7379],
    [0,        518.0534, 451.5029],
    [0,         0,        1.0000]
])# Your camera matrix
radial_distortion = np.array([-0.2658,0.0674,-0.0074])# Your radial distortion coefficients
tangential_distortion = 1e-3*np.array([0.8847, -0.7518])# Your tangential distortion coefficients
distortion_coefficients = np.array([radial_distortion[0],radial_distortion[1],tangential_distortion[0],tangential_distortion[1],radial_distortion[2]])# Your distortion coefficients

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None, camera_matrix)

    # Write the frame into the new file
    out.write(undistorted_frame)

    # Optional: Display the frame (can be commented out in final script)
    cv2.imshow('Frame', undistorted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
