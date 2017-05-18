-----------------------------------------------------------------------------------
-- Implementation of a RCNN for logo recognition for the 'Advanced Methods in
-- Machine Learning' course (SS2017)
--
-- This file contains testing and evaluation of the trained model
--
-- Authors: Tofunmi Ajayi, Ping Lu, Fisnik Mengjiqi, Roman Schindler
-----------------------------------------------------------------------------------
require 'torch'
require 'image'
require 'nn'
require 'optim'

require("training")
require("logoRec")


--TODO: This is for now just a dummy implementation that returns the cut-out logo
-- goal: read precomputed selective search output from txt-file
--   Returns: Region proposals for the image (nRegions x 3x32x32 DoubleTensor)
function selective_search(sample)
    local nb_regions = 1
    local regions = torch.DoubleTensor(nb_regions,3,32,32)
    if sample.label ~= 'no-logo' then
        r = extract_logo(sample)
    else
        local img_filename = images_path .. sample.label .. "/" .. sample.image_file
        r = image.load(img_filename,3,'byte')
    end
    r = image.scale(r,32,32)
    regions[1]:copy(r:double())
    return regions
end


-- Function to evaluate the model by classifying the test samples and computing
-- the accuracy of the classifications
--   Arguments: 
--   Returns:
-- TODO: computation of the prediction from the probabilites will change
--       when we have multiple regions instead of just one as in the current
--       implementation (due to the dummy output of the selective search function)
function evaluate_model(model, testset)
    -- load the test samples
    test_samples = read_data(testset)
    test_labels = torch.DoubleTensor(#test_samples)
    pred = torch.DoubleTensor(#test_samples)

    -- Classify each test image
    for i,sample in ipairs(test_samples) do
        -- get region proposals from selective search
        local regions = selective_search(sample)

        -- store true label of current test image
        test_labels[i] = labelMapping[sample.label]

        -- classify the region proposals and store prediction of current test image
        local probabilities = model:forward(regions)
        probabilities = probabilities:exp()
        conf,prediction = probabilities:max(1)
        --conf,prediction = probabilities:max(2)
        pred[i] = prediction
    end

    -- evaluate classifications
    local correct = torch.eq(pred:byte(),test_labels:byte()):sum()
    local accuracy = correct/test_labels:size(1)

    -- print results (sample-id, label, predicted label) and achieved accuracy
    for i = 1,test_labels:size(1) do
        local line = 'sample id: ' .. tostring(i) .. ', ' 
        line = line .. 'label: ' .. labelMapping[test_labels[i]] .. ', ' 
        line = line .. 'classified as: ' .. labelMapping[pred[i]] .. ' '
    --    line = line .. tostring(pred[i][1]) .. ' '
    --    for j = 1,probabilities:size(2) do
    --        line = line .. string.format("%.3f", probabilities[{i,j}]) .. ' '
    --    end
        print(line)
    end
    print('Accuracy: ' .. accuracy)
end

