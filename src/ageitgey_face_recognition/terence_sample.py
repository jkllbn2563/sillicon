#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import face_recognition.api as face_recognition
import cv2






img = cv2.imread('people.jpg')


face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model='hog')


for loc in face_locations:
	face = img[loc[0]:loc[2],loc[3]:loc[1],:]
	print (loc)
	cv2.imshow('x',face)
	cv2.waitKey(0)