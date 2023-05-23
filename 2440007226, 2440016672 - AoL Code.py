import cv2 as cv
import numpy as np
from keras.models import load_model

#declare the hand gesture categories
categories = ['call_me', 'fingers_crossed', 'okay', 'paper', 'peace', 'rock',
        'rock_on', 'scissor', 'thumbs', 'up']
#load the saved CNN model
model = load_model("vggmodel.h5")

#declare video input file
cap = cv.VideoCapture('HandGestureAll.mp4')

#declare video output file
out = cv.VideoWriter('vggoutput.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         30, (640,360))


while True:
    ret, frame = cap.read()
    if ret == False:
        break
    img = frame[0:720, 160:1120]
    #resize image
    img = cv.resize(img,(128, 128))
    #convert to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #apply gaussian blur
    img = cv.GaussianBlur(img,(5,5),0)
    #apply adaptive binary thresholding
    ret,img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    #predict using the CNN model
    result = model.predict(img.reshape(1, 128, 128, 3))
    result = list(enumerate(result.flatten()))

    result.sort(reverse = True, key = lambda x:x[1])
    print(categories[result[0][0]])

    #show results
    frame = cv.putText(frame, categories[result[0][0]], (10, 700), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 7)
    frame = cv.resize(frame, (640, 360))
    cv.imshow('frame', frame)
    #save to output file
    out.write(frame)

    if cv.waitKey(20) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()