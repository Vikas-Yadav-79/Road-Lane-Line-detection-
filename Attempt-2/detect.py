import cv2
import numpy as np

# Open the video file
video = cv2.VideoCapture('Videos_FOl\Video_6.mp4')

while True:
    # Read a frame from the video
    ret, frame = video.read()
    
    # Check if the frame was read successfully
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper thresholds for yellow color in HSV
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    
    # Create a mask to filter the yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply Canny edge detection to the masked frame
    edges = cv2.Canny(mask, 75, 150)

    if not ret:
        # If the video has ended, break out of the loop

        video = cv2.VideoCapture("Videos_FOl\Video_6.mp4")
        continue
    
    
    # Display the original frame
    cv2.imshow('Original Frame', frame)
    
    # Display the edges frame
    cv2.imshow('Edges', edges)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
