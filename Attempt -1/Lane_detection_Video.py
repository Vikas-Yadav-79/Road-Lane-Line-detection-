import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog


def ROI(img, vertices):
    mask = np.zeros_like(img)
    # Total_channel = img.shape[2]
    # match_mask_color=(255,) * Total_channel
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def draw_line(img, lines):
    img = np.copy(img)
    balnk_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(balnk_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

    img = cv2.addWeighted(img, 0.8, balnk_image, 1, 0.0)
    return img


# print(image.shape)


def Process(image):
    height = image.shape[0]
    Width = image.shape[1]


    # print(height)
    # print(Width)

    Region_OF_Interest_vertices = [  # (357, 595, 3)
        #(x1, y1),  # Top-left corner
        # (x2, y1),  # Top-right corner
        # (x2, y2),  # Bottom-right corner
        # (x1, y2)   # Bottom-left corner
        # (7, 818),  # img 4
        # (640, 417),
        # (1267, 842)
        # (355, 474),  # img 6
        # (511, 348),
        # (851, 449)
        # (Width/2 - height / 2 ,height*0.6 ),

        # Triangle
        (0, height),
        (Width / 2, height / 2),
        (Width, height)
        
        # rectangle
        # (0, height),
        # (height, 0),
        # (Width / 2, height / 2),
        # (0,Width/2)


        #rectangle
        # (0, 720),
        # (0, 0),
        # (1280, 0),
        # (1280,720)


        # updated roi
        # (400, 660),
        # (400, 450),
        # # (0, 0),
        # ((1280) /2 + 300, 450),

        # ( (1280) /2 + 300,660)






    

    ]
    Gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    canny_img = cv2.Canny(Gray_img, 100, 150)
    cropped_img = ROI(canny_img, np.array([Region_OF_Interest_vertices], np.int32))

    Hough_line = cv2.HoughLinesP(
        cropped_img,
        rho=6,
        theta=np.pi / 90,
        threshold=160,
        lines=np.array([]),
        minLineLength=70,
        maxLineGap=15,
    )
    image_with_lines = draw_line(image, Hough_line)

    return image_with_lines



root = tk.Tk()
root.withdraw()
input_path = filedialog.askopenfilename(title="Select input file")
root.destroy()  # Close the hidden window
cap = cv2.VideoCapture(input_path)



delay_between_frames = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = Process(frame)

    cv2.imshow("Frame", frame)
    cv2.waitKey(delay_between_frames)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
