"""
Live ESTMD demo

__author__: Dragonfly Project 2016 - Imperial College London
            ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)

"""

import cv2

from ESTMD.Estmd import Estmd
from Environment.WebCamHandler import WebCamHandler

webcam = WebCamHandler(run_id="Live_run", camera=0)
estmd = Estmd("Live_estmd", webcam.frame_dimensions, resize_factor=0.25)

while True:
    frame = webcam.read()
    green = webcam.green_filter(frame)
    result = estmd.process_frame(green)
    result = cv2.resize(result, webcam.frame_dimensions)
    cv2.imshow('Camera', frame)
    cv2.imshow('ESTMD', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.cap.release()
cv2.destroyAllWindows()
