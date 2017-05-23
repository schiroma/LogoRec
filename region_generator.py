import os
import skimage
import matplotlib.pyplot as plt
import selectivesearch
import numpy as np
from PIL import Image


dataset_path = "FlickrLogos-v2/"
bbox_path = "FlickrLogos-v2/classes/masks/"
images_path = "FlickrLogos-v2/classes/jpg/"
DIRACTORY_VAL= "regions/"


def create_folders():
	if not os.path.exists(DIRACTORY_VAL):
		os.makedirs(DIRACTORY_VAL)	
		

# -- Function that ....
# --   
# -- 
def read_data(data_file):
	samples = {} 
	with open(dataset_path + data_file, 'r') as f:		
		for line in f:			
			label, filename = line.split(",")
			if (label != "no-logo"):
				filename = filename.rstrip()			
				bbox = bbox_path + label + "/" + filename + ".bboxes.txt"
				sample_dic = {}
				sample_dic['label'] = label
				sample_dic['box'] = read_boundingboxes(bbox)	
				samples[filename] = sample_dic
				
	return samples
			  
# -- Function that ....
# --   
# -- 
def read_boundingboxes(bbox_file):
	boundingbox = {}
	with open(bbox_file,'r') as f: 
		header = True
		for line in f:
			if not header:
				x,y,w,h = line.split(" ")	
				boundingbox['x'] = int(x)
				boundingbox['y'] = int(y)
				boundingbox['w'] = int(w)
				boundingbox['h'] = int(h.rstrip())
			else:
				header=False
	return boundingbox	
# -- Function that generates annotated region proposals from an image using selective search
# --   
# --   
def generate_region_proposals(img_name,img_propery):
	
	label = img_propery['label']
	image_filename = images_path + label + "/" + img_name
	img = Image.open(image_filename)
	img_arr = np.asarray(img)
	img_lbl, regions = selectivesearch.selective_search(img_arr, scale=500, sigma=0.9, min_size=10)	
	candidates = set()

	# create a txt file where it saves generated regions for that image
	path_region = DIRACTORY_VAL + img_name[:-4] + ".txt"
	file = open(path_region,"w")
	file.write("label, x, y, width, height" + "\n")
	
	for r in regions:
		generated_sample_reg = {}
		# excluding same rectangle (with different segments)
		if r['rect'] in candidates:
			continue
		# excluding regions smaller than 2000 pixels
		if r['size'] < 1000:
			continue
		# distorted rects
		x, y, w, h = r['rect']
		if w / h > 5 or h / w > 5:
			continue	
		candidates.add(r['rect'])
		label = annotate_region(img_propery,r['rect'])	
		#save regions of a picture on a txt file
		x, y, w, h = r['rect']
		txt = label + ", " + str(x) + ", " + str(y) + ", " + str(w) + ", " + str(h)  + "\n"
		file.write(txt)
	file.close() 
	

# -- Function that annotates an image region with a label (the logo name or 'no logo')
# -- by comparing it with the ground truth bounding box of the logo 
# --   Returns: string representing the label of the input region
def annotate_region(prop, region):
	label = prop['label']
	box = prop['box']
	iou  = IOU(box,region)
	# annotate
	if (iou < 0.5):
		label = 'no-logo'

	return label  


def IOU(box,region):

	xB1, yB1, wB, hB = region
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(box['x'], xB1)
	yA = max(box['y'], yB1)
	xB = min(box['w'], wB)
	yB = min(box['h'], hB)
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (box['w'] - box['x'] + 1) * (box['h'] - box['y'] + 1)
	boxBArea = (wB - xB1 + 1) * (hB - yB1 + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area. +0.001 in case of division by zero.  
	iou = interArea / float(boxAArea + boxBArea - interArea + 0.001)
	return iou

	

# -- Function that ....
# -- samples  
# -- set : {1 for trainset.txt , 2 for trainvalset.txt }
def generate_training_data(samples):
	for img_name,img_property in samples.items(): 
		generate_region_proposals(img_name,img_property)
	

#-----------------------------------------------------------------


#-- Create output_python/ directory (folder)
create_folders() 

#-- Read the trainvalset images
training_data = read_data("trainvalset.txt")
generate_training_data(training_data)






