import cv2 as cv
import numpy as np
import time

def change_value(x):
  pass

capture = cv.VideoCapture(0, cv.CAP_DSHOW)

width=320
height=240

# Cam fov and focal length
hfov = 78
vfov = 43.3
focal_len_h = width / (2 * np.tan(hfov*np.pi/360) )
focal_len_v = height / (2 * np.tan(vfov*np.pi/360) )

capture.set(3, width) #width
capture.set(4, height) #height
capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25) #turn off auto-exposure
capture.set(cv.CAP_PROP_ZOOM, 10)

prev_time = 0
delta_x=0
delta_y=0

# cv.namedWindow("Adjustments")
# cv.createTrackbar("x", "Adjustments", 1, 10, change_value)

while True:
  start_time = time.time()
  success, img = capture.read()

  # x = cv.getTrackbarPos("x", "Adjustments")

  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  blur = cv.medianBlur(gray, 5)
  thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

  # Morph open 
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
  opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=3)

  cnts = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  for c in cnts:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
    area = cv.contourArea(c)
    if len(approx) > 5 and area > 1150 and area < 1450:
      print(area)
      ((x, y), r) = cv.minEnclosingCircle(c)
      cv.circle(img, (int(x), int(y)), int(r), (36, 255, 12), 2)
      cv.circle(img, (int(x), int(y)), 1, (255, 255, 255), 1)

      nx = round(x-160) 
      ny = round(120-y) 

      # change from pixels to degrees
      yaw = round(np.arctan(nx/focal_len_h) * 180 / np.pi, 2)
      pitch = round(np.arctan(ny/focal_len_v) * 180 / np.pi, 2)

      delta_y = round(23*np.tan(np.radians(pitch)), 2)
      delta_x = round(23*np.tan(np.radians(yaw)), 2)

      # Data overlay on img
      cv.putText(img, "y:" + str(int(delta_y)) + " x:" + str(int(delta_x)), 
                  (int(x), int(y)-30), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1)



  current_time = time.time()
  fps = 1/(current_time-prev_time)
  prev_time = current_time

  cv.putText(img, "FPS: {} Delta: {}".format(int(fps), round(np.sqrt((delta_x**2)+(delta_y**2)), 2)), (5,15), 
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


  cv.rectangle(img, ((width//2), (height//2) + 20), ((width//2), (height//2) - 20), (0,0,255), 2)
  cv.rectangle(img, ((width//2) + 20, (height//2)), ((width//2) - 20, (height//2)), (0,0,255), 2)

  cv.imshow('thresh', thresh)
  cv.imshow('opening', opening)
  cv.imshow('image', img)

  if cv.waitKey(20) & 0xFF==ord('f'):
    break

cv.destroyAllWindows()
