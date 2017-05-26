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
--require 'cutorch'
--require 'cunn'

require("logoRec")


-- Function that cuts out the gt-logo, crops it and returns it in a table
-- This function is maybe useful for analysing the model. It is not used in the
-- actual testing process of the final model
-- Note: i have changed some stuff and have not tested if it still works
--   in: sample (table) - a sample-table containing image-file, label and bbox
--   out: a table of images (the cropped logo of the sample)
function get_logo_regions(sample)
    local cropped_logos = {}
    if sample.label ~= 'no-logo' then
        r = crop_region(sample,sample.bbox[1])
    else
        local img_filename = images_path .. sample.label .. "/" .. sample.image_file
        r = image.load(img_filename,3,'byte')
    end
    r = image.scale(r,32,32)
    table.insert(cropped_logos,r:double())  
    return cropped_logos
end


-- Function that computes the region proposals, crops them and returns them in a table
--   in: sample (table) - a sample-table containing image-file, label and bbox
--   out: a table of images (the cropped regions from selective search)
function get_regions(sample)
    local cropped_regions = {}
    local regions = selective_search(sample)
    for j,reg in ipairs(regions) do
        local cropped_region = crop_region(sample,reg)
        cropped_region = image.scale(cropped_region,32,32)  
        table.insert(cropped_regions,cropped_region:double())       
    end
    return cropped_regions
end


-- TODO
-- Function that stores all predictions, confidences etc in a txt file for 
-- later analysis
--   in: labels (tensor) - the true labels of the classified samples
--       predictions (tensor) - the predicted labels of the classified samples
--       probabilities (tensor) - probabilities for each class
function save_results(labels,predictions,probabilities)
    for i = 1,labels:size(1) do
        line = tostring(test_labels[i]) .. ' ' 
        line = line .. tostring(pred[i]) .. ' '
    --    line = line .. tostring(pred[i][1]) .. ' '
        for j = 1,probabilities:size(2) do
            line = line .. string.format("%.3f", probabilities[{i,j}]) .. ' '
        end
        print(line)
    end

end


-- Function to evaluate the model by classifying the test samples and computing
-- the accuracy of the classifications
--   in: model - the trained cnn used for classification
--       testset (string) - name of the txt-file containing the image filenames 
--                          and corresponding logos of the test samples
function evaluate_model(model, testset)
    -- load the test samples
    tmp = read_data(testset)
    test_samples = {}
    for i,sample in ipairs(tmp) do
        if sample.label ~= 'no-logo' and sample.label ~= 'no' then
            table.insert(test_samples,sample)
        end
    end
    --test_samples = read_data(testset)
    test_labels = torch.DoubleTensor(#test_samples)
    pred = torch.DoubleTensor(#test_samples)
    conf = torch.DoubleTensor(#test_samples)

    -- classify each test image
    for i,sample in ipairs(test_samples) do
        -- get region proposals from selective search
        local regions = get_regions(sample)

        -- store true label of current test image
        test_labels[i] = labelMapping[sample.label]
        pred[i] = labelMapping['no-logo']
        max_conf = 0.0

        -- classify the region proposals and store prediction of current test image
        for j,reg in ipairs(regions) do
            -- classify region
            --reg = reg:cuda()
            local probabilities = model:forward(reg)
            probabilities = probabilities:exp()
            conf,prediction = probabilities:max(1)

            -- get most probable logo (i.e. second-most probable class if 'no-logo'
            -- was the most probable class) and its confidence
            if prediction[1] == labelMapping['no-logo'] then
                probabilities[prediction[1]] = 0.0
                conf,prediction2 = probabilities:max(1)

                -- keep this logo if probability above threshold, otherwise stay 
                -- with 'no-logo'
                if conf[1] > 0.15 then
                    prediction[1] = prediction2[1]
                end
            end

            -- if region classified with a logo, then keep this prediction for this 
            -- image and discard 'no-logo'. If different regions classified with 
            -- different logos, keep the one with highest confidence as final image pred.
            if prediction[1] ~= labelMapping['no-logo'] then
                if conf[1] > max_conf then
                    pred[i] = prediction
                    max_conf = conf[1]
                end
            end

            --print(prediction[1])
            --print(conf[1])
            --print(probabilities)
        end
        print(sample.label .. ' - ' .. labelMapping[pred[i]])
    end

    -- evaluate classifications
    local correct = torch.eq(pred:byte(),test_labels:byte()):sum()
    local accuracy = correct/test_labels:size(1)
    -- TODO confusion matrix
    -- TODO accuracy for each class
    -- TODO precision, recall

    -- print results (sample-id, label, predicted label) and achieved accuracy
    --for i = 1,test_labels:size(1) do
    --    local line = 'sample id: ' .. tostring(i) .. ', ' 
    --    line = line .. 'label: ' .. labelMapping[test_labels[i]] .. ', ' 
    --    line = line .. 'classified as: ' .. labelMapping[pred[i]] .. ' '
    --    line = line .. tostring(pred[i][1]) .. ' '
    --    for j = 1,probabilities:size(2) do
    --        line = line .. string.format("%.3f", probabilities[{i,j}]) .. ' '
    --    end
    --    print(line)
    --end
    print('Accuracy: ' .. accuracy)
    return accuracy
end

