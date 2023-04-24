from pie_data import PIE
import os
import cv2


if __name__ == '__main__':
	pie_path = os.getcwd()
	imdb = PIE(data_path=pie_path)
	# imdb.extract_and_save_images(extract_frame_type='annotated') # Uncomment out to split the video into images. Only images for annotated frames will be generated

	if 1:
		# Change the set_ids in the _get_vehicles function in pie_data.py to appropriate dataset
		train = imdb.generate_data_trajectory_sequence('train', seq_type='vehicle')
		# train = imdb.generate_data_trajectory_sequence('val', seq_type='vehicle')

		train_list = []
		for i in range(len(train['image'])):
			images = train['image'][i]
			for j in range(len(images)):
				image = images[j]
				frame_id = train['frame_id'][i][j]
				obj_class = ''.join(train['obj_class'][i][j])
				obj_type_list = train['obj_type'][i][j]
				obj_type_str_map = map(str, obj_type_list)
				obj_type_str = ''.join(obj_type_str_map)
				if obj_type_str == 'None':
					obj_type = 9999
				else:
					obj_type = int(obj_type_str)
				center = train['center_YOLO'][i][j]
				bbox = train['bbox_YOLO'][i][j]
				box = train['box'][i][j]
				# For testing
				# train_list.append([image, obj_type, obj_class, center[0], center[1], bbox[0], bbox[1], box])
				# train_list.append([image, obj_type, obj_class, center[0], center[1], bbox[0], bbox[1]])
				train_list.append([image, obj_type, obj_class, frame_id, int(box[0]), int(box[1]), int(box[2]), int(box[3])])

		if 0: # For testing
			# Reading an image in default mode
			image = cv2.imread(train_list[1001][0])

			coords = train_list[1001][7]

			# Start coordinate, here (5, 5)
			# represents the top left corner of rectangle
			start_point = (int(coords[0]), int(coords[1]))

			# Ending coordinate, here (220, 220)
			# represents the bottom right corner of rectangle
			end_point = (int(coords[2]), int(coords[3]))

			# Blue color in BGR
			color = (255, 0, 0)

			# Line thickness of 2 px
			thickness = 2

			# Using cv2.rectangle() method
			# Draw a rectangle with blue line borders of thickness of 2 px
			image = cv2.rectangle(image, start_point, end_point, color, thickness)

			# Displaying the image
			cv2.imwrite('val.png', image)

		if 1:
			for i in range(len(train_list)):
				if train_list[i][2] != 'vehicle' or train_list[i][1] > 0: # Select only cars
					continue

				temp = train_list[i][0].split('\\')
				isdir = os.path.isdir(temp[4])
				if not isdir:
					os.mkdir(temp[4])

				filename = os.path.join(temp[4], temp[5].split('.')[0] + '.txt') # If folders/files already exist, then delete them. Otherwise they will be appended!
				with open(filename, 'a') as f:
						f.write(" ".join(str(val) for val in train_list[i][2:]))
						f.write("\n")
