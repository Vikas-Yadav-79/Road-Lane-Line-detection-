
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define a function to create a region of interest (ROI) mask
def ROI(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Define a function to draw lines on an image with custom modifications
def draw_line(img, lines):
    img = np.copy(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Customize the line color and thickness
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)
            
            # Calculate the midpoint of the line
            x_mid = (x1 + x2) // 2
            y_mid = (y1 + y2) // 2
            
            # Annotate the line with text showing the midpoint coordinates
            coordinate_text = f"({x_mid}, {y_mid})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)  # White color
            font_thickness = 1
            cv2.putText(img, coordinate_text, (x_mid, y_mid), font, font_scale, font_color, font_thickness)
    
    return img

# Define the main processing function
def Process(image):
    height = image.shape[0]
    width = image.shape[1]

    # Define the region of interest vertices
    Region_OF_Interest_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    canny_img = cv2.Canny(gray_img, 100, 150)

    # Apply the ROI mask
    cropped_img = ROI(canny_img, np.array([Region_OF_Interest_vertices], np.int32))

    # Use the Hough Line Transform to detect lines
    Hough_line = cv2.HoughLinesP(cropped_img,
                                 rho=3,
                                 theta=np.pi / 60,
                                 threshold=160,
                                 lines=np.array([]),
                                 minLineLength=50,
                                 maxLineGap=25)
    
    # Draw the detected lines on the image with custom modifications
    image_with_lines = draw_line(image, Hough_line)

    return image_with_lines

# Open the video file for reading
cap = cv2.VideoCapture("Videos_FOl/Video_3.mp4")


delay_between_frames = 100

# Loop through the frames in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = Process(frame)

    # Display the processed frame
    cv2.imshow("Frame", frame)

    cv2.waitKey(delay_between_frames)
    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
