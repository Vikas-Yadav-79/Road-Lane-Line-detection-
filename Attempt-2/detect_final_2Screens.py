import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


# Open the video file
#  root = tk.Tk()
#     root.withdraw()  # Hide the main window
#     input_path = filedialog.askopenfilename(title="Select input file")
#     root.destroy()  # Close the hidden window


root = tk.Tk()
root.withdraw()
input_path = filedialog.askopenfilename(title="Select input file")
root.destroy()




video = cv2.VideoCapture(input_path)

while True:
    # Read a frame from the video
    ret, orig_frame = video.read()

    # Check if the frame was read successfully
    if not ret:
        # If the video has ended, break out of the loop

        video = cv2.VideoCapture(video)
        continue


    frame = cv2.GaussianBlur(orig_frame,(5,5),0)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper thresholds for yellow color in HSV
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])

    # Create a mask to filter the yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply Canny edge detection to the masked frame
    edges = cv2.Canny(mask, 65, 65)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50, minLineLength=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Display the original framew
    cv2.imshow("Original Frame", frame)

    # Display the edges frame
    cv2.imshow("Edges", edges)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
