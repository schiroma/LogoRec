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


-----------------------------------------------------------------------------------
-- Generate the training images 
-- arguments: 
--   - name of the folder to store the generated data (make sure a folder with  
--     this name exists, otherwise it will fail)
--   - filename with the trainingset information (trainset.txt or trainvalset.txt)
--   - true: with data augmentation (blurred images), false: without data augm.
-- Note: Can be commented out if they were generated before

-- generate the training regions without data augmentation
generate_training_data('preprocessed','trainvalset.txt',false)

-- generate the training regions and augment them with blurred images
generate_training_data('preprocessed-train','trainvalset.txt',true)


-----------------------------------------------------------------------------------
-- Train and save the model
-- arguments: 
--   - name of the folder with the generated training images 
--   - learning rate: e.g. 0.0005
--   - number of epochs: e.g. 15 
--   - batch size: e.g. 128 
--   - true: validation during training, false: no validation during training
-- Note: Can be commented out if it was trained before and saved to file

-- train a simple model without data augmentation
model1 = train_model('preprocessed', 0.0005, 15, 128, false)
torch.save('trained_model',model1)

-- train a more robust model that was augmented with blurred images
model2 = train_model('preprocessed-train', 0.0005, 15, 128, false)
torch.save('trained_model_2',model2)


-----------------------------------------------------------------------------------
-- Evaluate the model
-- arguments of evaluate_model: 
--   - trained model
--   - filename with the test-dataset information (testset.txt or testset-logosonly.txt)
--   - true: test distorted images, false: test undistorted images
--   - true: output predictions and store results in files, false: no output/files 
--   - 1: an id for the test (used for the name of the output file with the results)

-- load the trained models from file (if they were already trained before)
model1 = torch.load('trained_model')
model2 = torch.load('trained_model_2')

-- experiment 1: test simple model with undistorted images (only test images with logos) 
exp_id = 1
evaluate_model(model1, 'testset-logosonly.txt', false, true, exp_id)

-- experiment 2: test simple model with undistorted images (entire test set) 
exp_id = 2
evaluate_model(model1, 'testset.txt', false, true, exp_id)

-- experiment 3: test simple model with distorted images (entire test set) 
exp_id = 3
evaluate_model(model1, 'testset.txt', true, true, exp_id)

-- experiment 4: test robust model with undistorted images (entire test set) 
exp_id = 4
evaluate_model(model2, 'testset.txt', false, true, exp_id)

-- experiment 5: test robust model with distorted images (entire test set) 
exp_id = 5
evaluate_model(model2, 'testset.txt', true, true, exp_id)


