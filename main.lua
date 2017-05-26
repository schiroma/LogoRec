-----------------------------------------------------------------------------------
-- Implementation of a RCNN for logo recognition for the 'Advanced Methods in
-- Machine Learning' course (SS2017)
--
-- This file contains the main function
--
-- Authors: Tofunmi Ajayi, Ping Lu, Fisnik Mengjiqi, Roman Schindler
-----------------------------------------------------------------------------------
require("training")
require("testing")
require("logoRec")

-- Generate the training images 
-- arguments: 
--   - name of the folder to store the generated data (make sure a folder with  
--     this name exists)
--   - filename with the trainingset information (trainset.txt or trainvalset.txt)
--   - true: with data augmentation (blurred images), false: without data augm.
-- Note: Can be commented out if they were generated before
generate_training_data('preprocessed','trainvalset.txt',false)

-- Train and save the model
-- arguments: 
--   - name of the folder with the generated training images ('processed')
--   - learning rate: e.g. 0.0005
--   - number of epochs: e.g. 14 
--   - batch size: e.g. 128 
--   - true: validation during training, false: no validation during training
-- Note: Can be commented out if it was trained before and saved to file
model = train_model('preprocessed', 0.0005, 15, 128, false)
torch.save('trained_model',model)

-- Evaluate the model
-- arguments: 
--   - trained model
--   - filename with the test-dataset information (testset.txt or testset-logosonly.txt)
--   - true: test distorted images, false: test undistorted images
--model = torch.load('trained_model')
evaluate_model(model, 'valset.txt', false)
--evaluate_model(model, 'valset-small.txt', false)
