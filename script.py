import cv2
import pandas as pd
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor

#CONSTS
data_train_path = './'
file_name = 'Видео для теста-Обрезка 01.MOV'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
windowName="Webcam Live video feed"
iterFrames = 4
itera = 1
thresh = 75
kern = 1

model = XGBClassifier()
model.load_model('best_model.json')

model_puls = XGBRegressor()
model_puls.load_model('xgb_model.json')


#FUNCS
def detect_face(frame):
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    return faces[0]

def safe_div(x,y): # so we don't crash so often
    if y==0: return 0
    return x/y

def rescale_frame(frame, percent=25):  # make the video windows a bit smaller
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


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
    xywh[i][0] = xIM 
    xywh[i][1] = yIM
    xywh[i][2] = wIM
    xywh[i][3] = hIM
    xywh[i][4] = wIM + hIM

print(xywh[np.unravel_index(np.argmax(xywh[:, 4], axis=None), xywh[:, 4].shape)[0]])
xIM1, yIM1, wIM1, hIM1, wh = xywh[np.unravel_index(np.argmax(xywh[:, 4], axis=None), xywh[:, 4].shape)[0]]

closing_s = []

showLive=True
while(showLive):
    ret, frame=videocapture.read()
    try:
        frame = frame[yIM1:yIM1+3*hIM1, xIM1-wIM1:xIM1+2*wIM1]
    except:
        break
    

    frame_resize = rescale_frame(frame)
    if not ret:
        print("cannot capture the frame")
        exit()
   
    ret,thresh1 = cv2.threshold(frame_resize,thresh,255,cv2.THRESH_BINARY) 
    thresh1 = (255-thresh1)
    kernel = np.ones((kern,kern),np.uint8)
    
    dilation = cv2.dilate(thresh1, kernel, iterations=itera)
    erosion = cv2.erode(dilation,kernel,iterations = itera)

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  
    closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
    
    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    closing = cv2.cvtColor(closing,cv2.COLOR_GRAY2RGB)
    
    areas = [] #list to hold all areas

    for contour in contours:
      ar = cv2.contourArea(contour)
      areas.append(ar)

    if len(areas) > 0:
        max_area = max(areas)
        max_area_index = areas.index(max_area)  # index of the list element with largest area

        cnt = contours[max_area_index - 1] # largest area contour is usually the viewing window itself, why?

        #cv2.drawContours(closing, [cnt], 0, (0,0,255), 1)
        
        closing[closing.sum(axis=2)<255*3] = 0
        closing_s += [closing]
        cv2.imshow('', closing)
        if cv2.waitKey(30)>=0:
            showLive=False

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

#plt.plot(delts)
#plt.show()

sub_df = pd.DataFrame({'delts' : delts})

sub_mean = np.exp(sub_df.mean().item()/10)
sub_std = np.exp(sub_df.std().item()/10)
sub_max = np.exp(sub_df.max().item()/10)
sub_median = np.exp(sub_df.median().item()/10)

df = pd.DataFrame({'mean' : [sub_mean], 'std' : [sub_std], 'max' : [sub_max], 'median' : [sub_median]})
y_pred = model.predict(df.drop(['max', 'mean'], axis=1))

df['freq'] = [int(y_pred[0])]
puls = model_puls.predict(df)
if int(y_pred[0]) > 0:
    puls[0] += 25
    
print(int(y_pred[0]), puls)
color = [0, 0, 0]
if int(y_pred[0]) == 0:
    color = [0, 255, 0]
elif int(y_pred[0]) == 1:
    color = [0, 165, 255]
else:
    color = [0, 0, 255]

videocapture = cv2.VideoCapture(data_train_path + file_name)
while videocapture.isOpened:
    ret, frame = videocapture.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.rectangle(frame, (xIM1, yIM1), (xIM1+wIM1, yIM1+hIM1), color, 2)
    cv2.putText(frame, str(int(puls[0])), (xIM1, yIM1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    cv2.imshow('', frame)
    if cv2.waitKey(30)>=0:
        break

videocapture.release()
cv2.destroyAllWindows()