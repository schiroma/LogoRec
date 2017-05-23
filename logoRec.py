import os
import skimage
import matplotlib.pyplot as plt
import selectivesearch
import numpy as np
from PIL import Image


dataset_path = "FlickrLogos-v2/"
bbox_path = "FlickrLogos-v2/classes/masks/"
images_path = "FlickrLogos-v2/classes/jpg/"



def create_folder():
	directory = 'output_python'
	if not os.path.exists(directory):
	    os.makedirs(directory)

# -- Function that ....
# --   
# -- 
def read_data(data_file):
	samples = {} 
	i=1
	with open(dataset_path + data_file, 'r') as f:		
		for line in f:	
			i+=1		
			label, filename = line.split(",")
			filename = filename.rstrip()			
			bbox = bbox_path + label + "/" + filename + ".bboxes.txt"
			sample_dic = {}
			sample_dic['label'] = label
			sample_dic['box'] = read_boundingboxes(bbox)	
			samples[filename] = sample_dic
			#----------------------------
			#--reads only two training images, then it stops i>3. 
			#--Since, right now per one image, selective search is generating hunders of regions/images that have to 
			#--be croped and saved, and it takes time and eats memory.
			if i>3:
				break
			#----------------------------									
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
	
# -- Function that ....
# --   
# -- 
# It takes to arguments: Image file name, and properties of a image (label, and boudingboxes values x,y,width,height)
def extract_logo(img_name,prop):
	
	label = prop['label']
	box = prop['box']
	img_filename  = images_path + label + '/' + img_name
	img = Image.open(img_filename)
	logo = crop_image(img,box)
	return logo

# -- Function that ....
# --   
# -- 
def crop_image(img,bbox):

	x1 = bbox['x']
	y1 = bbox['y']
	x2 = x1 + bbox['w']
	y2 = y1 + bbox['h']
	return img.crop((x1,y1,x2,y2))

# -- Function that generates annotated region proposals from an image using selective search
# --   
# --   
def generate_region_proposals(img_name,img_propery):
	
	label = img_propery['label']
	image_filename = images_path + label + "/" + img_name
	img = Image.open(image_filename)
	img_arr = np.asarray(img)
	
	img_lbl, regions = selectivesearch.selective_search(img_arr, scale=500, sigma=0.9, min_size=10)	

	genereted_regisons = {}
	i=1
	for r in regions:
		generated_sample_reg = {}
		
		#-----------------------------------------------------------
		# TODO:  
		#-- FIX the issue with multi black regisons (generated img)*
		#-- FIX the issue with same/similar regions (genereted img)*
		
		# excluding same rectangle (with different segments)
		if r['rect'] in genereted_regisons:
			continue
		# excluding regions smaller than 2000 pixels
		if r['size'] < 2000:
			continue
		# distorted rects
		x, y, w, h = r['rect']
		if w / h > 1.2 or h / w > 1.2:
			continue
		#-----------------------------------------------------------

		#TODO: Create annotate_region method
		#label = annotate_region(sample,region)	
		#TODO fix the values x y w h as in croped_img method 
		reg = img.crop((x,y,w,h))		
		#TODO: Change the key name 'logo', because the generated sample region is just a croped part of the image 
		generated_sample_reg['logo'] = reg
		#TODO: 'no label' should be replaced with label, after the implementaiton of annotate_region
		generated_sample_reg['label'] = 'no logo'
		genereted_regisons[i] = generated_sample_reg
		i+=1
	return genereted_regisons
	

# -- Function that annotates an image region with a label (the logo name or 'no logo')
# -- by comparing it with the ground truth bounding box of the logo 
# --   Arguments: sample (table) - a sample-table containing image-file, label and bbox
# --              region (tensor) - 1x4 vector (x/y of upper left corner, width and height)
# --   Returns: string representing the label of the input region
# --def annotate_region(sample, region):
#     -- TODO
#     -- compute IoU (intersection over union) between logo-bbox and region
#     -- ...
#     --
#     -- label = 'nologo'
#     -- if (IoU > 0.5) then
#     --     label = sample.label
#     -- end
#     -- return label  



# -- Function that ....
# --   
# -- 
def generate_training_data(samples):
	training_data = {}
	i = 1
	for img_name,img_property in samples.items(): 
		generated_sample = {}
		logo = extract_logo(img_name,img_property)
		generated_sample['logo'] = logo
		generated_sample['label'] = img_property['label']
		# img_key used as key (name) per generated sample , [:-4] removes '.jpg' from the img_name 
		img_key = img_name[:-4]
		training_data[img_key] = generated_sample

		generated_regions = generate_region_proposals(img_name,img_property)
		for i,region in generated_regions.items():
			logo_key = img_key + '_' + str(i)
			generated_region = {}
			generated_region['logo'] = region['logo']
			generated_region['label'] = region['label']
			training_data[logo_key] = generated_region

	# TO DO: Refactoing, craete a method for this task
	for i,sample in training_data.items():
		logo = sample['logo']
		sample['logo'] = logo.resize((32,32), Image.ANTIALIAS)
		
	return training_data

# -- Function that ....
# --   
# -- 
def save_images(img_name, img):
	img.save('output_python/'+ img_name)



#------------------------------------------------------------------------------

  
#-- Read the training images
training_samples = read_data("trainset.txt")

#-- Generate the training data from the training images
training_data = generate_training_data(training_samples)

#-- Create output_python/ directory (folder)
create_folder()

#-- Save the generated images to folder
for i,v in training_data.items():
	logo = v['logo']
	save_images( 'tr_' + str(i) + '.jpg' ,logo)




		
	



