import cv2
import pandas as pd
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import matplotlib.pyplot as plt

#CONSTS
data_train_path = './data/noisySvin/Выдосы нарезанные/пульс 116-90 21 1/'
file_name = 'пульс 116-90-Обрезка 16.MOV'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
windowName="Webcam Live video feed"
iterFrames = 5

#FUNCS
def detect_face(frame):
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    return faces[0]

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def safe_div(x,y): # so we don't crash so often
    if y==0: return 0
    return x/y


def rescale_frame(frame, percent=25):  # make the video windows a bit smaller
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# in a terminal
# python -m pip install --user opencv-contrib-python numpy scipy matplotlib ipython jupyter pandas sympy nose
# using cam built-in to computer
# using IP camera address from my mobile phone, with Android 'IP Webcam' app over WiFi
# videocapture=cv2.VideoCapture("http://xxx.xxx.xxx.xxx:8080/video")
# Sliders to adjust image
# https://medium.com/@manivannan_data/set-trackbar-on-image-using-opencv-python-58c57fbee1ee
#cv2.createTrackbar("threshold", windowName, 75, 255, nothing)
#cv2.createTrackbar("kernel", windowName, 5, 30, nothing)
#cv2.createTrackbar("iterations", windowName, 1, 10, nothing)


videocapture = cv2.VideoCapture(data_train_path + file_name)
if not videocapture.isOpened():
    print("can't open camera")
    exit()

xywh = np.ndarray((iterFrames, 5), dtype=int)
for i in range(iterFrames):
    ret, frame=videocapture.read()
    
    #FRAME NORMALIZING
    frame = cv2.normalize(frame, None, 50, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    xIM, yIM, wIM, hIM = detect_face(frame)

    cv2.rectangle(frame, (xIM, yIM), (xIM+wIM,yIM+hIM), (255, 0, 0), 2)
    cv2.imshow('', frame)
    if cv2.waitKey(30)>=0:
        showLive=False
    #===============

    xywh[i][0] = xIM 
    xywh[i][1] = yIM
    xywh[i][2] = wIM
    xywh[i][3] = hIM
    xywh[i][4] = wIM + hIM

print(xywh[np.unravel_index(np.argmax(xywh[:, 4], axis=None), xywh[:, 4].shape)[0]])
xIM1, yIM1, wIM1, hIM1, wh = xywh[np.unravel_index(np.argmax(xywh[:, 4], axis=None), xywh[:, 4].shape)[0]]


#cv2.namedWindow(windowName)
#cv2.createTrackbar("threshold", windowName, 75, 255, nothing)
#cv2.createTrackbar("kernel", windowName, 5, 30, nothing)
#cv2.createTrackbar("iterations", windowName, 1, 10, nothing)

closing_s = []

showLive=True
while(showLive):
    ret, frame=videocapture.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv.rectangle(frame, (x-w, 0), (x+2*w, 480), (255, 0, 0), 2)
    try:
        frame = frame[yIM1:, xIM1-wIM1:xIM1+2*wIM1]
    except:
        break
    

    frame_resize = rescale_frame(frame)
    if not ret:
        print("cannot capture the frame")
        exit()
   
    #thresh= cv2.getTrackbarPos("threshold", windowName) 
    thresh = 75
    ret,thresh1 = cv2.threshold(frame_resize,thresh,255,cv2.THRESH_BINARY) 
    thresh1 = (255-thresh1)
    #cv2.imwrite(name, imagem)
    
    #kern=cv2.getTrackbarPos("kernel", windowName) 
    kern = 1
    kernel = np.ones((kern,kern),np.uint8) # square image kernel used for erosion
    
    #itera=cv2.getTrackbarPos("iterations", windowName) 
    itera = 1
    dilation = cv2.dilate(thresh1, kernel, iterations=itera)
    erosion = cv2.erode(dilation,kernel,iterations = itera) # refines all edges in the binary image

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  
    closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
    
    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # find contours with simple approximation cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE

    closing = cv2.cvtColor(closing,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(closing, contours, -1, (128,255,0), 1)
    
    # focus on only the largest outline by area
    areas = [] #list to hold all areas

    for contour in contours:
      ar = cv2.contourArea(contour)
      areas.append(ar)

    if len(areas) > 0:
        max_area = max(areas)
        max_area_index = areas.index(max_area)  # index of the list element with largest area

        cnt = contours[max_area_index - 1] # largest area contour is usually the viewing window itself, why?

        cv2.drawContours(closing, [cnt], 0, (0,0,255), 1)
        
        closing[closing.sum(axis=2)<255*3] = 0
        closing_s += [closing]
        cv2.imshow('', closing)
        if cv2.waitKey(30)>=0:
            showLive=False
        
videocapture.release()
cv2.destroyAllWindows()

closing_delts = []
print(type(closing_s[0]))

delts = np.zeros(len(closing_s)-1, dtype=int)
for i in range(1, len(closing_s)):
    cl_dl = closing_s[i] - closing_s[i-1]
    closing_delts += [cl_dl]
    cv2.imshow('cl_dl', cl_dl)
    if cv2.waitKey(30)>=0:
        showLive=False

    delts[i-1] = cl_dl.sum()/(255*3)

plt.plot(delts)
plt.show()

df = pd.DataFrame()
df['delts'] = delts
df.to_csv(data_train_path + 'delts3.csv') 