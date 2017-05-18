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
-- Note: We will do this generation with python. hence this function call will 
-- be removed later (I leave it for now so this script runs without error)
generate_training_data("trainset.txt")

-- Train the model
model = train_model('output', 0.001, 20, 32)

-- Evaluate the model
evaluate_model(model, 'testset-logosonly.txt')
