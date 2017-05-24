# LogoRec

Torch implementation of a RCNN for logo recognition

Notes:
- Put the dataset (i.e. the folder 'FlickrLogos-v2') in the same directory as the lua file
- Create an empty folder named 'preprocessed' in the same directory as the lua file
- Rename folder 'hp' in FlickrLogos-v2/classes/masks to 'HP'
- The current implementation loads the training data, generates the region proposals and stores them in the folder 'preprocessed'. The images in this folder are then used to train the network. It then uses the logo-images in the validationset for testing.
- Before using the lua implementation you have to generate the selective search output for each image with the python implementation. Since this takes dozens of hours I uploaded those that I generated already (trainset and valset-logosonly) on dropbox:
    https://www.dropbox.com/s/dn0cq945wzshs6z/regions_trainvallogo.tar.gz?dl=0
  You need these files in a folder named 'regions' and it has to be in the same directory as the lua files.
- Generating the training data for all training samples (trainset and valset-logosonly) also takes some time. Therefore I also uploaded this data on dropbox
    https://www.dropbox.com/s/l7qezsidix9j8px/preprocessed_trainvallogo.tar.gz?dl=0
  You need these files in a folder named 'preprocessed' and it has to be in the same directory as the lua files

# LogoRec Python implementation

Notes: 
- Put the dataset (i.e. the folder 'FlickrLogos-v2') in the same directory as the python (or lua) file
- Install selective search:
	$ pip install selectivesearch
- GitHub repo of selective saerch: https://github.com/AlpacaDB/selectivesearch
- The current implementation loads the training data, cuts out the logos from the 
training images, selective search algorithm generates regions in each traning image, crop them and save in output_python folder (automaticlly created)

