# sillicon
### emotion recognition
```
cd ageitgey_face_recognition
python terence_sample_ros_server.py
cd ..
python visualize.py
python client_sillicon.py
```
### object_detection
```
python object_server.py 
python object_client.py 
(if yor do not want to visualize the image please comment the 
vis_util.visualize_boxes_and_labels_on_image_array(
	  image_copy,
	  #output_dict['detection_boxes'],
	  #output_dict['detection_classes'],
	  #output_dict['detection_scores'],
	  filtered_dict['detection_boxes'],
	  filtered_dict['detection_classes'],
	  filtered_dict['detection_scores'],
	  category_index,
	  instance_masks=output_dict.get('detection_masks'),
	  use_normalized_coordinates=True,
	  min_score_thresh=.0,
	  line_thickness=8)
)
```
