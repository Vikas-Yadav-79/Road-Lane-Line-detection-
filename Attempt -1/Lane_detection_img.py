import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("Images_FOl\img4.jpeg")

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def ROI(img,vertices):
    mask=np.zeros_like(img)
    # Total_channel = img.shape[2]
    # match_mask_color=(255,) * Total_channel
    match_mask_color  = 255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image=cv2.bitwise_and(img,mask)

    return masked_image



def draw_line(img,lines):

    img = np.copy(img)
    balnk_image=np.zeros((img.shape[0], img.shape[1], 3) , dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(balnk_image,(x1,y1),(x2,y2) , (0,255,0) , thickness=3)


    img = cv2.addWeighted(img,0.8,balnk_image,1,0.0)
    return img


# print(image.shape)

height = image.shape[0]
Width = image.shape[1]

Region_OF_Interest_vertices =[   #(357, 595, 3)
#    (355,474),  # img 6 
#     (511,348),
#     (851,449) 

    (0, height),
        (Width / 2, height / 2),
        (Width, height),
    
     (7,818),#img 4
    (640,417),
    (1267,842)
 ]





Gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

canny_img = cv2.Canny(Gray_img,100,50)
cropped_img = ROI(canny_img,np.array([Region_OF_Interest_vertices],np.int32))


Hough_line = cv2.HoughLinesP(cropped_img,
                             rho=6 ,
                             theta=np.pi / 90,
                             threshold=160,
                             lines=np.array([]),
                             minLineLength=50,
                             maxLineGap=25)


if Hough_line is not None:  
    image_with_lines = draw_line(image, Hough_line)
    plt.imshow(image_with_lines)
else:
    print("No lines detected")

plt.show()