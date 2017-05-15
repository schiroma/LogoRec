# LogoRec

Torch implementation of a RCNN for logo recognition

Notes:
- Put the dataset (i.e. the folder 'FlickrLogos-v2') in the same directory as the lua file
- Create an empty folder named 'output' in the same directory as the lua file
- The current implementation loads the training data, cuts out the logos from the 
training images and stores these cropped logo images to the output folder



# LogoRec Python implementation

Notes: 
- Put the dataset (i.e. the folder 'FlickrLogos-v2') in the same directory as the python (or lua) file
- Install selective search
	$ pip install selectivesearch
- GitHub repo of selective saerch: https://github.com/AlpacaDB/selectivesearch
- The current implementation loads the training data, cuts out the logos from the 
training images, selective search algorithm generates regions in each traning image, crop them and save in output_python folder (automaticlly created)

