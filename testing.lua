-----------------------------------------------------------------------------------
-- Implementation of a RCNN for logo recognition for the 'Advanced Methods in
-- Machine Learning' course (SS2017)
--
-- This file contains testing and evaluation of the trained model
--
-- Authors: Tofunmi Ajayi, Ping Lu, Fisnik Mengjiqi, Roman Schindler
-----------------------------------------------------------------------------------
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
--       distorted (bool) - if true, distoreted (blurred) versions of the images are used
--   out: a table of images (the cropped regions from selective search)
function get_regions(sample, distorted)
    local cropped_regions = {}
    local regions = selective_search(sample)
    local img = load_image(sample)
    if distorted then
        img = blur_image(img)
    end
    for j,reg in ipairs(regions) do
        local cropped_region = crop_region(img,reg)
        cropped_region = image.scale(cropped_region,32,32)  
        table.insert(cropped_regions,cropped_region:double())       
    end
    return cropped_regions
end


-- Function that stores all results in a txt file for later analysis
--   in: id (int) - the test id (used for the name of the file we write)
--       acc (double) - maximized accuracy 
--       t (double) - threshold that leads to best accuracy
--       cm - confusion matrix
--       labels (tensor) - the true labels of the classified samples
--       predictions (tensor) - the predicted labels of the classified samples
--       probabilities (tensor) - confidence of each prediction
function save_results(id,acc,t,cm,labels,predictions,probabilities)
    -- concatenate labels, predictions and probabilites into one matrix
    results = torch.cat(labels,predictions,2)
    results = torch.cat(results,probabilities,2)

    -- write the data to a file
    file = io.open('results' .. tostring(id) .. '.txt','w')
    file:write('best accuracy: ' .. tostring(acc))
    file:write('\n')
    file:write('best threshold: ' .. tostring(t))
    file:write('\n\n')
    file:write(tostring(cm))
    file:write('\n\n')
    file:write(tostring(results))
    file:close()
end


-- Function that computes the accuracy for different thresholds on the confidence.
-- The output can be used to find the the threshold that leads to best accuracy
--   in: labels (tensor) - contains the true label of each samples
--       pred (tensor) - contains the most probable logo of each sample
--       conf (tensor) - contains the confidence of the most probable logo of each image
--   out: the accuracies achieved with different thersholds
--        the corresponding thresholds
function evaluate_predictions(labels,pred,conf)    
    local accuracies = torch.Tensor(#conf):fill(0)
    local thresholds = torch.Tensor(#conf):fill(0)
    
    -- sort labels and predictions and confidences by confidence
    local conf,idx = torch.sort(conf)
    local pred = pred:index(1,idx)
    local labels = labels:index(1,idx)

    -- get number of samples and add a 0 at the beginning of the conf-vector
    local nb_samples = conf:size()[1]
    local conf_padded = torch.cat(torch.Tensor({0}),conf,1)

    for i=1,nb_samples do
        -- turn the predictions with conf below current threshold to 'no-logo'
        local thresholded_pred = torch.Tensor(#conf):fill(labelMapping['no-logo'])
        thresholded_pred[{{-i,-1}}] = pred[{{-i,-1}}]

        -- compute and store accuracy achieved with current threshold
        local correct = torch.eq(thresholded_pred,labels):sum()
        accuracies[i] = correct/nb_samples
        thresholds[i] = (conf_padded[-i]+conf_padded[-i-1])/2
    end

    return accuracies,thresholds
end


-- Function to apply a threshold on the predictions, i.e. set all all predictions
-- with confidence below this threshold to zero
function threshold_predictions(pred,conf,threshold)
    for i=1,pred:size()[1] do
        if conf[i] < threshold then
            pred[i] = labelMapping['no-logo']
        end
    end
    return pred
end


-- Function to evaluate the model by classifying the test samples and computing
-- the accuracy of the classifications
--   in: model - the trained cnn used for classification
--       testset (string) - name of the txt-file containing the image filenames 
--                          and corresponding logos of the test samples
--       distorted (bool) - if true, distoreted (blurred) images are used for testing
--       output (bool) - if true, predictions are output, and results written to files
--       test_id (int) - an id of the test (used for the filename of the results)
--   out: the accuracy of the predictions
function evaluate_model(model, testset, distorted, output, test_id)
    -- load the test samples
    local test_samples = read_data(testset,false)
    local test_labels = torch.DoubleTensor(#test_samples)
    local pred = torch.DoubleTensor(#test_samples)
    local conf = torch.DoubleTensor(#test_samples)

    -- classify each test image
    for i,sample in ipairs(test_samples) do
        -- get region proposals from selective search
        local regions = get_regions(sample,distorted)

        -- store true label of current test image
        test_labels[i] = labelMapping[sample.label]

        -- classify the region proposals and keep most probable logo found in any
        -- of the regions (and the corresponding confidence)
        max_conf = 0.0
        for j,reg in ipairs(regions) do
            -- classify region
            --reg = reg:cuda()
            local probabilities = model:forward(reg)
            probabilities = probabilities:exp()
            confidence,prediction = probabilities:max(1)

            -- get most probable logo (i.e. second-most probable class if 'no-logo'
            -- was the most probable class) and its confidence
            if prediction[1] == labelMapping['no-logo'] then
                probabilities[prediction[1]] = 0.0
                confidence,prediction = probabilities:max(1)
            end

            -- keep prediction with highest confidence
            if confidence[1] > max_conf then
                pred[i] = prediction[1]
                conf[i] = confidence[1]
                max_conf = confidence[1]
            end
        end

        -- show the prediction (just most probable logo. not thresholded yet)
        if output then
            print(sample.label .. ' - ' .. labelMapping[pred[i]])
        end
    end

    -- evaluate classifications: find threshold that leads to best accuracy
    -- and apply this threshold to the predictions
    local accuracies,thresholds = evaluate_predictions(test_labels,pred,conf)
    local top_acc,idx = accuracies:max(1)
    local top_threshold = thresholds[idx[1]]
    local top_predictions = threshold_predictions(pred,conf,top_threshold)

    if output then
        -- compute confusion matrix
        local cm = optim.ConfusionMatrix(#classes,classes)
        cm:batchAdd(top_predictions,test_labels)

        -- plot accuracy per threshold and store it as png file
        gnuplot.pngfigure('acc-threshold' .. tostring(test_id) .. '.png')
        gnuplot.plot(thresholds,accuracies)
        gnuplot.axis({0,1,0,1})
        gnuplot.xlabel('threshold')
        gnuplot.ylabel('accuracy')
        gnuplot.plotflush()

        -- save all the results in a txt file
        save_results(test_id, top_acc[1], top_threshold, cm, test_labels, top_predictions, conf)
        -- show achieved accuracy and corresponding threshold
        print('Accuracy: ' .. tostring(top_acc[1]) .. ', best threshold: ' .. tostring(top_threshold))
    end
    return top_acc
end

