#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import face_recognition.api as face_recognition
import cv2

import rospy
from face_detection_service.srv import *
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

def face_detection_body(img):
	face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model='hog')
	bounding_boxes = []
	for loc in face_locations:
		#bounding_boxes.extend([float(x) for x in loc])
		bounding_boxes.extend(loc)
	return bounding_boxes


def handle_face_detection(req):
	print ('start to handle face detection service')
	cv_image = bridge.imgmsg_to_cv2(req.robot_view, desired_encoding="passthrough")
	bounding_boxes = face_detection_body(cv_image)
	return faceDetectionV3Response(bounding_boxes)

rospy.init_node('face_detection_server')
s_face_detection = rospy.Service('face_detection_service', faceDetectionV3, handle_face_detection)
print ("face detection server init done.")
rospy.spin()