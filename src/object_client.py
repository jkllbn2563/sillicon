#!/usr/bin/env python
import rospy
from sillicon_pkg.srv import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from face_detection_service.srv import *
import cv2
img=Image()
bridge = CvBridge()

def rgb_callback(image):
	global img
	img=image

def bbox_calculation():	
	global img
	try:
		t = detection_client(img)
		#print (t)
		object_name = t.detection_results
		bbox = t.bounding_boxes_pixels
		#object_name,bbox=detection_client(img)
		#print(bbox.data)
		#if bbox.data==(0.0,0.0,0.0,0.0):
			#print("There is no %s on the table"%object_name)
		return object_name,bbox

	except rospy.ServiceException, e:
		rospy.logwarn("service call failed:%s"%e)



if __name__=='__main__':
	rospy.init_node('client_object_node',anonymous=True)
	#rospy.Subscriber("/usb_cam/image_raw",Image,rgb_callback)
	#rospy.Subscriber("/c1/camera/depth/points",PointCloud2,point_cloud_callback)
	frame=cv2.imread("banana.jpg")

	img = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
	rospy.wait_for_service('object_detection')
	detection_client=rospy.ServiceProxy('object_detection',objectDetectionV2)
	#rate=rospy.Rate(1)
	obj_name,bbox_data=bbox_calculation()
	print("There are %s"%obj_name,"and there are in",bbox_data)


