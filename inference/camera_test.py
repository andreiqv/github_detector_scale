#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Using TF for inference, and TensorRT for compress a graph.

import sys
import os
import argparse

import numpy as np
from PIL import Image
import time
import timer

from time import sleep
import io
from picamera import PiCamera
from picamera.array import PiRGBArray

camera = PiCamera()
stream = io.BytesIO()

camera.start_preview()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

INPUT_SIZE = [3, 128, 128]

timer.timer('predictions.eval')	

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image - this array
	# will be 3D, representing the width, height, and # of channels
	image = frame.array
 
	# show the frame
	#cv2.imshow("Frame", image)
	print(type(image))
	timer.timer()
	#key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	#if key == ord("q"):
	#	break
	#time.sleep(1)


for i in range(10):

	camera.capture(stream, format='jpeg')
	stream.seek(0)
	image = Image.open(stream)
	shape = tuple(INPUT_SIZE[1:])
	image = image.resize(shape, Image.ANTIALIAS)
	image_arr = np.array(image, dtype=np.float32) / 255.0				

	pred_val = predictions.eval(feed_dict={input_: [image_arr]})
	print(pred_val)
	timer.timer()
	#time_res.append(0)
	#print('index={0}, label={1}'.format(index, label))

	camera.stop_preview()	
	print(camera.resolution)

	#print('mean time = {0}'.format(np.mean(time_res)))

	#return index


