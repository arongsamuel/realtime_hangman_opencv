import cv2
import numpy as np
import joblib
from skimage.feature import hog
from hangman_function import hangman
f=open("words.txt","r")
words = f.read().splitlines()
font = cv2.FONT_HERSHEY_SIMPLEX
letters='abcdefghijklmnopqrstuvwxyz'
clf = joblib.load("trained_model.pkl")
flag=0

#This determines the scale of ROI(Region of Interest), which is bounded by the white rectangle.
hscale=0.325 
wscale=0.25  

cap = cv2.VideoCapture(2)
while(True):    
    # Capture frame by frame from video source 0
    wordsb=words.copy()
    blank={}
    fill=[]
    ret, frame = cap.read()
    org_img = frame

    org_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    x1,y1=org_gray.shape[1],org_gray.shape[0]
    cv2.rectangle(org_img, (int(x1*wscale), int(y1*hscale)), (int((1-wscale)*x1), int((1-hscale)*y1)), (255, 255, 255), 1)
    org_gray = org_gray[int(y1*hscale):int((1-hscale)*y1),int(x1*wscale):int((1-wscale)*x1)]    
    morphStructure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(org_gray, cv2.MORPH_GRADIENT, morphStructure)
    ret2, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morphStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 7))  
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morphStructure)
    _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if w/h>3:
            blank[x]='_'
            fill.append([x,y])
        else:                       
            if h < 5 or w < 5:
                continue
            crop_closed = closed[y:y+h, x:x+w]
            r = cv2.countNonZero(crop_closed)/(w * h)
            if r > 0.20:
                try:                    
                    #cv2.rectangle(org_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#The values for thresholding changes with the lighting in the room
#I got good results with (127,255), change this according to your conditions
                    _, img = cv2.threshold(org_gray[y:y+h, x:x+w], 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    imout = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
                    imout = cv2.dilate(imout, (3, 3))              
                    roi_hog_fd = hog(cv2.bitwise_not(imout), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
                    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
                    blank[x]=letters[int(nbr[0])]
#Uncomment the line below to see if the recognition of the given letters are correct                            
                    #cv2.putText(org_img,letters[int(nbr[0])].capitalize(),(int(wscale*x1)+x,int(hscale*y1)+y), font, 1, (255,0,0), 2, cv2.LINE_AA)
#Uncomment the line below to see if the detection and cleaning of the given letters are correct                 
                    #org_img[int(hscale*y1)+y:int(hscale*y1)+y+h, int(wscale*x1)+x:int(wscale*x1)+x+w]=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                    
                except:
                    "skip"
    input=','.join([blank[i] for i in sorted(blank)]).replace(',', '')
    print(input)
    output=hangman(input,wordsb)
    if output:
        cv2.putText(org_img,str(flag+1)+" of "+str(len(output))+" solutions, press C to cycle through the solutions",(int((wscale/2)*x1),int((hscale/2)*y1)), font,0.5, (255,255,255), 1, cv2.LINE_AA)    
        try:                
            for position in fill:                
                letout=output[flag][sorted(blank).index(position[0])]
                cv2.putText(org_img,letout.capitalize(),(int(wscale*x1)+position[0],int(hscale*y1)+position[1]), font, 1.4, (255,255,255), 2, cv2.LINE_AA)
            
        except:
            "skip"
    cv2.putText(org_img,"Pess Q to quit",(int((wscale)*x1),int(3*hscale*y1)), font,0.75, (255,255,255), 1, cv2.LINE_AA)                           
    cv2.imshow('frame',org_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(99) & 0xFF == ord('c'):
        if output:
            print(output[flag])            
        flag+=1
        flag=flag%len(output)        
cap.release()
cv2.destroyAllWindows()
