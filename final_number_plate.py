from darkflow.net.build import TFNet
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import pytesseract

options = {"pbLoad": "yolo-plate.pb", "metaLoad": "yolo-plate.meta",}
yoloPlate = TFNet(options)
options = {"pbLoad": "yolo-character.pb", "metaLoad": "yolo-character.meta"}
yoloCharacter = TFNet(options)
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    characterRecognition = load_model('character_recognition.h5')
img = cv2.imread('swift.jpg',0)
#enter the image
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def firstCrop(img, predictions):
    predictions.sort(key=lambda x: x.get('confidence'))
    xtop = predictions[-1].get('topleft').get('x')
    ytop = predictions[-1].get('topleft').get('y')
    xbottom = predictions[-1].get('bottomright').get('x')
    ybottom = predictions[-1].get('bottomright').get('y')
    firstCrop = img[ytop:ybottom, xtop:xbottom]
    return firstCrop

def secondCrop(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)    
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]        
        x,y,w,h = cv2.boundingRect(cnt)
        secondCrop = img[y:y+h,x:x+w]
    else:
        secondCrop = img
    return secondCrop

def yoloCrop(predictions,img):
    positions = []
    x_top=[]
    x_bottom=[]
    y_top=[]
    y_bottom=[]
    for i in predictions:
        if i.get("confidence")>0.20:
            xtop = i.get('topleft').get('x')
            positions.append(xtop)
            x_top.append(xtop)
            ytop = i.get('topleft').get('y')
            y_top.append(ytop)
            xbottom = i.get('bottomright').get('x')
            x_bottom.append(xbottom)
            ybottom = i.get('bottomright').get('y')
            y_bottom.append(ybottom)
    abc=img[min(y_top)-10:max(y_bottom)+10,min(x_top)-5:max(x_bottom)+10]    
    return abc

predictions = yoloPlate.return_predict(img)
firstCropImg = firstCrop(img, predictions)
img = secondCrop(firstCropImg)
img1=img.copy()
predictions = yoloCharacter.return_predict(img)
img=yoloCrop(predictions,img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.medianBlur(gray, 3)
gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)
print(gray.shape[0],gray.shape[1])
height_abc=gray.shape[0]
width_abc=gray.shape[1]
diff_h=img1.shape[0]-height_abc
diff_w=img1.shape[1]-width_abc
if diff_w % 2 == 0:
    x1 = np.ones(shape=(height_abc, diff_w//2))
    x2 = x1
else:
    x1 = np.ones(shape=(height_abc, diff_w//2))
    x2 = np.ones(shape=(height_abc, (diff_w//2)+1))

gray= np.concatenate((x1, gray, x2), axis=1)
x1 = np.ones(shape=(diff_h//2, gray.shape[1]))
x2 = x1
gray= np.concatenate((x1, gray, x2), axis=0)
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
text = pytesseract.image_to_string(gray)

cv2.imshow('Second crop plate',gray) 
cv2.waitKey(0)
cv2.destroyAllWindows()