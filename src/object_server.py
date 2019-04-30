#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import rospy
from sillicon_pkg.srv import *
from cv_bridge import CvBridge,CvBridgeError

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict




def server_srv():
	rospy.init_node('detection_server',anonymous=True)
	s=rospy.Service('object_detection',objectDetectionV2,handle_function)
	rospy.loginfo('Ready to caculate the bbox')
	rospy.spin()


def handle_function(req):
	bridge = CvBridge()
	#print("fuck")
	image_np = bridge.imgmsg_to_cv2(req.robot_view, desired_encoding="passthrough")
	#image_np = bridge.imgmsg_to_cv2(req.robot_view, desired_encoding="rgb8")

	#image_np = bridge.imgmsg_to_cv2(req.robot_view, "rgb8")
	image_np_expanded = np.expand_dims(image_np, axis=0)
	output_dict = run_inference_for_single_image(image_np, detection_graph)
	filtered_dict = {'detection_boxes':[],'detection_classes':[],'detection_scores':[]}
	#filtered_banana_dict = {'detection_boxes':[],'detection_classes':[],'detection_scores':[]}
	#filtered_mouse_dict = {'detection_boxes':[],'detection_classes':[],'detection_scores':[]}
	#filtered_bottle_dict = {'detection_boxes':[],'detection_classes':[],'detection_scores':[]}
	"""
	for i in range(len(output_dict['detection_classes'])):
	  if (output_dict['detection_classes'][i]==52):
	      #filtered_banana_dict['detection_boxes'].append(output_dict['detection_boxes'][i])
	      #filtered_banana_dict['detection_classes'].append(output_dict['detection_classes'][i])
	      #filtered_banana_dict['detection_scores'].append(output_dict['detection_scores'][i])
	      filtered_dict['detection_boxes'].append(output_dict['detection_boxes'][i])

	      filtered_dict['detection_classes'].append(output_dict['detection_classes'][i])
	      filtered_dict['detection_scores'].append(output_dict['detection_scores'][i])

	      break
	for i in range(len(output_dict['detection_classes'])):
	  if (output_dict['detection_classes'][i]==74):
	      #filtered_mouse_dict['detection_boxes'].append(output_dict['detection_boxes'][i])
	      #filtered_mouse_dict['detection_classes'].append(output_dict['detection_classes'][i])
	      #filtered_mouse_dict['detection_scores'].append(output_dict['detection_scores'][i])
	      filtered_dict['detection_boxes'].append(output_dict['detection_boxes'][i])
	      filtered_dict['detection_classes'].append(output_dict['detection_classes'][i])
	      filtered_dict['detection_scores'].append(output_dict['detection_scores'][i])
	      break          
	for i in range(len(output_dict['detection_classes'])):
	  if (output_dict['detection_classes'][i]==44):
	      #filtered_bottle_dict['detection_boxes'].append(output_dict['detection_boxes'][i])
	      #filtered_bottle_dict['detection_classes'].append(output_dict['detection_classes'][i])
	      #filtered_bottle_dict['detection_scores'].append(output_dict['detection_scores'][i])
	      filtered_dict['detection_boxes'].append(output_dict['detection_boxes'][i])
	      filtered_dict['detection_classes'].append(output_dict['detection_classes'][i])
	      filtered_dict['detection_scores'].append(output_dict['detection_scores'][i])
	      break                      

	"""
	filtered_dict['detection_boxes'] = np.array(filtered_dict['detection_boxes'])
	#filtered_banana_dict['detection_boxes'] = np.array(filtered_banana_dict['detection_boxes'])
	#filtered_mouse_dict['detection_boxes'] = np.array(filtered_mouse_dict['detection_boxes'])
	#filtered_bottle_dict['detection_boxes'] = np.array(filtered_bottle_dict['detection_boxes'])
	
	filtered_dict['detection_classes'] = np.array(filtered_dict['detection_classes'])
	
	filtered_dict['detection_scores'] = np.array(filtered_dict['detection_scores'])
	# Visualization of the results of a detection.
	image_copy = np.copy(image_np)##It is amazing for me
	
	vis_util.visualize_boxes_and_labels_on_image_array(
	  image_copy,
	  output_dict['detection_boxes'],
	  output_dict['detection_classes'],
	  output_dict['detection_scores'],
	  #filtered_dict['detection_boxes'],
	  #filtered_dict['detection_classes'],
	  #filtered_dict['detection_scores'],
	  category_index,
	  instance_masks=output_dict.get('detection_masks'),
	  use_normalized_coordinates=True,
	  min_score_thresh=0.1,
	  line_thickness=8)
	print(image_np.size)

	#plt.figure(figsize=IMAGE_SIZE)
	#plt.imshow(image_copy)
	#plt.show()
	print(type(category_index))
	print(category_index)	
	img = Image.fromarray(image_np, 'RGB')
	
	high,width=img.size
	print("image size is",high,width)
	
	#object_dict={52:"banana",74:"mouse",44:"bottle"}
	#object_dict.setdefault("nonregist")
	#print(filtered_dict['detection_classes'])
	print(output_dict['detection_classes'])

	#print(type(filtered_dict['detection_classes']))
	string=" "
	#object_list=[object_dict[a] for a in list(filtered_dict['detection_classes'])]
	object_list=[category_index[a]["name"] for a in list(output_dict['detection_classes'])]

	object_list=string.join(object_list)
	#print(type(object_list))
	#object_list=[string.join(a) for a in object_dict[list(filtered_dict['detection_classes'])] ]
	
	print(object_list)
	#bbox_data=list(filtered_dict['detection_boxes'])
	bbox_data=list(output_dict['detection_boxes'])

	#print(bbox_data)

	
	if (output_dict['detection_boxes'].shape[0]!=0):
		bbox_data_list=[list(a) for a in bbox_data]
		#print("bbox",bbox_data_list)
		bbox_data=[]
		#print(len(bbox_data_list))
		for a in range(len(bbox_data_list)):
			bbox_data.extend(bbox_data_list.pop(0))
		print(bbox_data)

		for a in range(len(bbox_data)):
			if a%2==0:
				bbox_data[a]=bbox_data[a]*high
			else:
				bbox_data[a]=bbox_data[a]*width
		print(bbox_data)



	


	return objectDetectionV2Response(object_list,bbox_data)
  

if __name__== '__main__':
	# What model to download.
	MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
	MODEL_FILE = MODEL_NAME + '.tar.gz'
	DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

	"""
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())

	"""
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')

	#category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
	NUM_CLASSES=90
	label_map= label_map_util.load_labelmap(PATH_TO_LABELS)
	categories=label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	IMAGE_SIZE = (12, 8)

	server_srv()



    		

