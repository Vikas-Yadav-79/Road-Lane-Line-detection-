"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
import tkinter as tk
from tkinter import filedialog

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        # out_img = self.lanelines.plot(out_img)
        return out_img
# //python main.py --video Advanced-Lane-Lines\challenge_video.mp4 Advanced-Lane-Lines\.mp4

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():
    # Create a simple GUI for selecting input and output paths
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    input_path = filedialog.askopenfilename(title="Select input file")
    root.destroy()  # Close the hidden window

    if not input_path :
        print("Input and output paths are required.")
        return

    output_path_vid = "Output_videos\output_video11.mp4"

    output_path_img = "Output_images\out_1"

    findLaneLines = FindLaneLines()
    findLaneLines.process_video(input_path, output_path_vid)

    # findLaneLines.process_image(input_path,output_path_img)1


if __name__ == "__main__":
    main()