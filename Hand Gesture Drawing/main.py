import cv2
import numpy as np
import HandTrackingModule as htm

dectector = htm.handDetector()
draw_colour = (0,0,0)
brush_size = 10
eraser_size = 30
img_canvas = np.zeros((720,1280,3),np.uint8)

video = cv2.VideoCapture(0)

while True:
    sucess,frame = video.read()
    frame = cv2.flip(frame,1)
#draw 
    frame = cv2.resize(frame,(1280,720))
    cv2.rectangle(frame,(0,220),(1280,0),(0,0,0),-1)
    cv2.rectangle(frame,(40,200),(330,20),(22,0,225),-1)
    cv2.rectangle(frame,(340,200),(630,20),(255,22,0),-1)
    cv2.rectangle(frame,(640,200),(930,20),(0,225,20),-1)
    cv2.rectangle(frame,(940,200),(1230,20),(255,255,255),-1)
    cv2.putText(frame,'Eraser',(970,110),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)


# find hand cordinates
    img = dectector.findHands(frame)
    lmlist = dectector.findPosition(frame)
    
    if len(lmlist) !=0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

#find which finger is up
        fingers = dectector.fingersUp()
        if fingers[1] and fingers[2]:
            xp,yp = 0,0

            if y1<200:
                if 40<x1<330:
                    draw_colour = (22,0,225)
                elif 340<x1<630:
                    draw_colour = (255,22,0)
                elif 640<x1<930:
                    draw_colour = (0,225,20)
                elif 940<x1<1230:
                    draw_colour = (0,0,0)

            cv2.rectangle(frame,(x1,y1),(x2,y2),draw_colour,-1)

        if fingers[1] and not fingers[2]:
            cv2.circle(img,(x1,y1),10,draw_colour,-1)

            if xp == 0 and yp == 0:
                xp = x1
                yp = y1
            
            if draw_colour == (0,0,0):
              cv2.line(img,(xp,yp),(x1,y1),draw_colour,thickness=eraser_size)
              cv2.line(img_canvas,(xp,yp),(x1,y1),draw_colour,thickness=eraser_size)
            
            else:
                cv2.line(img,(xp,yp),(x1,y1),draw_colour,thickness=brush_size)
                cv2.line(img_canvas,(xp,yp),(x1,y1),draw_colour,thickness=brush_size)

            xp,yp = x1,y1

    img_gray = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)
    _,img_inv = cv2.threshold(img_gray,20,255,cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img,img_inv)
    img = cv2.bitwise_or(img,img_canvas)

    img = cv2.addWeighted(img,1,img_canvas,0.5,0)


    cv2.imshow('Video',img)
    if cv2.waitKey(1) & 0XFF==27:
        break
video.release()
cv2.destroyAllWindows()