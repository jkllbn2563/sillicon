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




def rgb_callback(image):
	global img,receive_first_image
	img=image

def bbox_calculation():	
	global img
	try:
		
		object_name,bbox=detection_client(img)
		#print(bbox.data)
		if bbox.data==(0.0,0.0,0.0,0.0):
			print("There is no %s on the table"%object_name)
		return object_name,bbox

	except rospy.ServiceException, e:
		rospy.logwarn("service call failed:%s"%e)



def face_calculation():	
	global img
	image_ros = bridge.cv2_to_imgmsg(img, encoding="8UC3")

	try:
				
		result=emotion_classfication_client(image_ros)
				
		return result
	except rospy.ServiceException, e:
		rospy.logwarn("service call failed:%s"%e)

if __name__=='__main__':
	rospy.init_node('client_node',anonymous=True)
	#rospy.Subscriber("/usb_cam/image_raw",Image,rgb_callback)
	#rospy.Subscriber("/c1/camera/depth/points",PointCloud2,point_cloud_callback)
	#rospy.wait_for_service('object_detection')

	rospy.wait_for_service('emotion_classfication_service')
	emotion_classfication_client=rospy.ServiceProxy('emotion_classfication_service',emotionRecognition)
	#detection_client=rospy.ServiceProxy('object_detection',objectDetectionV2)


	rate=rospy.Rate(1)
	
	if rospy.is_shutdown():
		exit(-1)

	rospy.wait_for_service('face_detection_service')

	bridge = CvBridge()


	cap = cv2.VideoCapture(0)

	# Capture frame-by-frame
	ret, frame = cap.read()

	frame = cv2.imread('ageitgey_face_recognition/people.jpg')

	image_message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
	face_detection_client = rospy.ServiceProxy('face_detection_service', faceDetectionV3)
	resp = face_detection_client(image_message)
	bboxes = resp.bounding_boxes_pixels
	print (bboxes)
	bboxes = [[x for x in bboxes[i*4:(i+1)*4]] for i in range(len(bboxes)/4)]
	global img
	for bbox in bboxes:
		face = frame[bbox[0]:bbox[2],bbox[3]:bbox[1],:]
		
		img=face

		emotion_result=face_calculation()
		
		print("the emotion is %s" % emotion_result.face_emotion)
		cv2.imshow('frame',face)
		cv2.waitKey(0)

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


	rospy.spin()






	