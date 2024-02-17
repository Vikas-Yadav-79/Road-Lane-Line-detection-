import cv2
import matplotlib.pyplot as plt
import numpy as np




image = cv2.imread("Images_FOl\img4.jpeg")

# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


Gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

canny_img = cv2.Canny(Gray_img,100,200)


h_i = cv2.HoughLinesP(canny_img,
                      rho=6,
                      theta=np.pi/60,
                      threshold=150,lines=np.array([]),
                      minLineLength=50,
                      maxLineGap=15)
for line in h_i:
    for x1,y1,x2,y2 in line:
         cv2.line(image,(x1,y1),(x2,y2) , (0,255,0) , thickness=3)




plt.imshow(image)

plt.show()