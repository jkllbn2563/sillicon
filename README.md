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
### object_detection(This si the code for finetune model to replace the label_map)
```
NUM_CLASSES=90
	label_map= label_map_util.load_labelmap(PATH_TO_LABELS)
	categories=label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	IMAGE_SIZE = (12, 8)
```
