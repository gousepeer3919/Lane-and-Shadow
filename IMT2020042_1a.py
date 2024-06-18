#import libraries
import numpy as np
import cv2 as cv
#reading images
image = cv.imread('road1.jpeg',1)
#converting image into grayscale image
gray_scale_image = cv.imread('road1.jpeg',0)
cv.imwrite('gray_scale.jpeg',gray_scale_image)
#converting into gaussian blurr image
gaussian_blur_image = cv.GaussianBlur(gray_scale_image, (5, 5), 2)
cv.imwrite('blurr_image.jpeg',gaussian_blur_image)
#using canny edge
canny_edge=cv.Canny(gaussian_blur_image,150,250)
cv.imwrite('canny_edge1.jpeg',canny_edge)

r=int(np.ceil(np.sqrt(canny_edge.shape[0]**2+canny_edge.shape[1]**2)))

#accumalator
acc=np.zeros((2*r,180), dtype=np.uint64)
for x in range(canny_edge.shape[0]):
    for y in range(canny_edge.shape[1]):
        if(canny_edge[x,y]>0):
            for k in range(180):
                rou = y*np.cos(k*np.pi/180) + x*np.sin(k*np.pi/180)
                acc[int(rou+r),k] += 1

n=6

for i in range(4):
    index=np.unravel_index(np.argmax(acc, axis=None), acc.shape)
    print(index[0]-r, " ", index[1]*np.pi/180)
    a=np.cos(index[1]*np.pi/180)
    b=np.sin(index[1]*np.pi/180)
    x=a*(index[0]-r)
    y=b*(index[0]-r)
    x3 = int(x + 1000*(-b))
    y3 = int(y + 1000*(a))
    x4 = int(x - 1000*(-b))
    y4 = int(y - 1000*(a))
    cv.line(image,(x3,y3),(x4,y4),(0,0,255),2)
    x1,x2,y1,y2 = 0,0,0,0
    if(index[0]-n<0):
        x1=0
    else:
        x1=index[0]-n
    if(index[0]+n>acc.shape[0]):
        x2=acc.shape[0]
    else:
        x2=index[0]+n
    if(index[1]-n<0):
        y1=0
    else:
        y1=index[1]-n
    if(index[1]+n>acc.shape[1]):
        y2=acc.shape[1]
    else:
        y2=index[1]+n
    for i in range(x1,x2+1):
        for j in range(y1,y2+1):
            acc[i, j]=0


cv.imwrite('lane_lines1.jpeg',image)
