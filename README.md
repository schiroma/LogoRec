# LogoRec

Torch implementation of a RCNN for logo recognition

Notes:
- Put the dataset (i.e. the folder 'FlickrLogos-v2') in the same directory as the lua file
- Create an empty folder named 'output' in the same directory as the lua file
- Rename folder 'hp' in FlickrLogos-v2/classes/masks to 'HP'
- The current implementation loads the training data, cuts out the logos from the training images and stores these cropped logo images to the output folder. These cropped images are then used to train the cnn (takes just some seconds since there are only 320 training images for now. It will be more when we include selective search and jittering). For testing I cut out the logos from the test images and classify them using the trained model. Accuracy is around 60-70%. This test is however a simplification. At the end we cannot cut out the logos from the test images and classify them. We'll have to use selective search.



# LogoRec Python implementation

Notes: 
- Put the dataset (i.e. the folder 'FlickrLogos-v2') in the same directory as the python (or lua) file
- Install selective search:
	$ pip install selectivesearch
- GitHub repo of selective saerch: https://github.com/AlpacaDB/selectivesearch
- The current implementation loads the training data, cuts out the logos from the 
training images, selective search algorithm generates regions in each traning image, crop them and save in output_python folder (automaticlly created)

