import cv2

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Open the video file
video_path = 'project/trymefirst_lisbon.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('lisbon.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there are no frames left

    # Convert frame to grayscale (SIFT works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints for visualization
    frame_with_keypoints = cv2.drawKeypoints(gray, keypoints, frame, color=(0,255,0))

    # Write the frame into the file 'output_video.avi'
    out.write(frame_with_keypoints)

    # Display the frame
    cv2.imshow('Frame with SIFT Features', frame_with_keypoints)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

print(f"Processed {frame_count} frames")

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
