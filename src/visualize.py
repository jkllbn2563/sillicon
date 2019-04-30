#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import rospy
from robot_arm.srv import *
from cv_bridge import CvBridge,CvBridgeError
from sillicon_pkg.srv import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
from sensor_msgs.msg import Image as Image2

#img=Image()
#raw0_image=Image2()
cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
def server_srv():
	rospy.init_node('detection_server',anonymous=True)
	s=rospy.Service('emotion_classfication_service',emotionRecognition,handle_function)
	rospy.loginfo('Ready to caculate the emotion')
	rospy.spin()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def handle_function(req):
	
	print("fuck")
	bridge = CvBridge()
	image_np = bridge.imgmsg_to_cv2(req.face, desired_encoding="passthrough")
	#image_np = bridge.imgmsg_to_cv2(req.face, "rgb8")
	#print("apple")
	
	#print("bbanana")
	#raw_image=Image.fromarray(image_np,"RGB")
	#raw_image = req.face

	#raw_img = io.imread('images/1.jpg')
	#gray = rgb2gray(raw_img)
	#gray = resize(gray, (48,48), mode='reflect').astype(np.uint8)

	#img = gray[:, :, np.newaxis]

	#img = np.concatenate((img, img, img), axis=2)
	#img = Image.fromarray(img)
	img=Image.fromarray(image_np)
	inputs = transform_test(img)


	
	ncrops, c, h, w = np.shape(inputs)

	inputs = inputs.view(-1, c, h, w)
	#inputs = inputs.cuda()#if tou want to use gpu
	inputs = Variable(inputs, volatile=True)
	outputs = net(inputs)

	outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

	score = F.softmax(outputs_avg)
	_, predicted = torch.max(outputs_avg.data, 0)

	#plt.rcParams['figure.figsize'] = (13.5,5.5)
	#axes=plt.subplot(1, 3, 1)
	#plt.imshow(raw_img)
	#plt.xlabel('Input Image', fontsize=16)
	#axes.set_xticks([])
	#axes.set_yticks([])
	#plt.tight_layout()


	#plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

	#plt.subplot(1, 3, 2)
	ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
	width = 0.4       # the width of the bars: can also be len(x) sequence
	color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
	"""
	for i in range(len(class_names)):
	    plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
	plt.title("Classification results ",fontsize=20)
	plt.xlabel(" Expression Category ",fontsize=16)
	plt.ylabel(" Classification Score ",fontsize=16)
	plt.xticks(ind, class_names, rotation=45, fontsize=14)
	"""
	#axes=plt.subplot(1, 3, 3)
	emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
	#plt.imshow(emojis_img)
	#plt.xlabel('Emoji Expression', fontsize=16)
	#axes.set_xticks([])
	#axes.set_yticks([])
	#plt.tight_layout()
	# show emojis

	#plt.show()
	#plt.savefig(os.path.join('images/results/l.png'))
	#plt.close()

	print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
	return str(class_names[int(predicted.cpu().numpy())])
	  

if __name__== '__main__':
	class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	net = VGG('VGG19')
	checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'),map_location=lambda storage,loc:storage)#for cpu
	net.load_state_dict(checkpoint['net'])
	#net.cuda()#if you want to use gpu
	net.eval()
	#print("hi")
	server_srv()


