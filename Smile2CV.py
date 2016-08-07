from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import time
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages/Adafruit_MCP4725-1.0.1-py2.7.egg')
sys.path.append('/usr/local/lib/python2.7/dist-packages/Adafruit_GPIO-1.0.0-py2.7.egg')
sys.path.append('/usr/local/lib/python2.7/dist-packages/Adafruit_PureIO-0.2.0-py2.7.egg')
sys.path.append('/usr/local/lib/python2.7/dist-packages/Adafruit_LED_Backpack-1.8.0-py2.7.egg')
import Adafruit_MCP4725
from PIL import Image
from PIL import ImageDraw
from Adafruit_LED_Backpack import Matrix8x8

ledNothing = 0
ledFace = 1
ledSmile = 2
drawDebug = False

def drawArray(display, array):
    display.begin()
    display.clear()
    for x in range(8):
        for y in range(8):
            display.set_pixel(y, x, array[x][y])
    display.write_display()
    

def drawSmile(display):
    face = [
    [0,0,1,1,1,1,0,0],
    [0,1,0,0,0,0,1,0],
    [1,0,1,0,0,1,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,1,0,0,1,0,1],
    [1,0,0,1,1,0,0,1],
    [0,1,0,0,0,0,1,0],
    [0,0,1,1,1,1,0,0]]
    drawArray(display, face)

def drawNothing(display):
    nothing = [
    [1,1,1,1,1,1,1,1],
    [1,1,0,0,0,0,1,1],
    [1,0,1,0,0,1,0,1],
    [1,0,0,1,1,0,0,1],
    [1,0,0,1,1,0,0,1],
    [1,0,1,0,0,1,0,1],
    [1,1,0,0,0,0,1,1],
    [1,1,1,1,1,1,1,1]]
    drawArray(display, nothing)

def drawFace(display):
    face = [
    [0,0,1,1,1,1,0,0],
    [0,1,0,0,0,0,1,0],
    [1,0,1,0,0,1,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,1],
    [0,1,0,0,0,0,1,0],
    [0,0,1,1,1,1,0,0]]
    drawArray(display, face)

def drawLED(display, toDraw):
    if toDraw is ledNothing:
        drawNothing(display)
    elif toDraw is ledFace:
        drawFace(display)
    elif toDraw is ledSmile:
        drawSmile(display)

display = Matrix8x8.Matrix8x8()
display.begin()
display.clear()
display.set_pixel(0, 0, 1)
display.write_display()

ledToDraw = 0
print("Starting camera")
# webcam = VideoStream(src=0).start()
webcam = cv2.VideoCapture(0)
display.clear()
display.set_pixel(1, 0, 1)
display.write_display()

print("Starting dac")
dac = Adafruit_MCP4725.MCP4725(address=0x60, busnum=1)
display.clear()
display.set_pixel(2, 0, 1)
display.write_display()

face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_smile.xml')

display.clear()
display.set_pixel(3, 0, 1)
display.write_display()

sF = 1.2
max_neighbors = -1
min_neighbors = -1
intensity = 0

webcam.set(3, 320.0)
webcam.set(4, 240.0)
display.clear()
display.set_pixel(4, 0, 1)
display.write_display()

print webcam.get(3)
print webcam.get(4)

smileToFaceRatio = 0
drawFace(display)
while True:
    drawLED(display, ledToDraw)
    ledToDraw = ledNothing
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=sF,
                                          minNeighbors=8,
                                          minSize=(55, 55))

    smileToFaceRatio = 0.85 * smileToFaceRatio
    for(x, y, w, h) in faces:
        ledToDraw = ledFace
        faceArea = w*h
        if drawDebug:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray,
                                               scaleFactor=1.2,
                                               minNeighbors=22,
                                               minSize=(25, 25))
        for(x, y, w, h) in smile:
          ledToDraw = ledSmile
          print "Found ", len(smile), " smiles"
          cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
          smileToFaceRatio = float(w*h) / faceArea
          minRatio = 0.02
          maxRatio = 0.24
          print "Raw ratio: ", smileToFaceRatio
          smileToFaceRatio = max(smileToFaceRatio - minRatio, 0)
          smileToFaceRatio = min(smileToFaceRatio / (maxRatio - minRatio), 1)
          print "Clamped ratio: ", smileToFaceRatio
      
        smile_neighbors = len(smile)
        if min_neighbors == -1:
            min_neighbors = smile_neighbors
        max_neighbors = max(max_neighbors, smile_neighbors)

    intensity = smileToFaceRatio
    if drawDebug:
        rows = frame.shape[0]
        cols = frame.shape[1]
        rect_height = round(rows * smileToFaceRatio)
        cv2.rectangle(frame, (0, int(rows)), (int(cols/10), int(rows - rect_height)), (0, 255, 0), -1)

  
    dac.set_voltage(int(4096 * intensity), True)
    if drawDebug:
        cv2.imshow('Smile Detector', frame)
    #time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()

