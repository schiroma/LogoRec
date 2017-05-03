# LogoRec

Torch implementation of a RCNN for logo recognition

Notes:
- Put the dataset (i.e. the folder 'FlickrLogos-v2') in the same directory as the lua file
- Create an empty folder named 'output' in the same directory as the lua file
- The current implementation loads the training data, cuts out the logos from the 
training images and stores these cropped logo images to the output folder