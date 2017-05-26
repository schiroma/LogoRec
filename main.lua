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
-- Note: Can be commented out if they were generated before
generate_training_data("trainset.txt")
--generate_training_data("trainvalset.txt")

-- Train the model
-- Note: Can be commented out if it was trained before and saved to file
model = train_model('preprocessed', 0.0005, 14, 128, false)
torch.save('trained_model',model)

-- Evaluate the model
--model = torch.load('trained_model')
--evaluate_model(model, 'testset-logosonly.txt')
evaluate_model(model, 'valset.txt')

