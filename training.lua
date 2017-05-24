-----------------------------------------------------------------------------------
-- Implementation of a RCNN for logo recognition for the 'Advanced Methods in
-- Machine Learning' course (SS2017)
--
-- This file contains the training of the model
--
-- Authors: Tofunmi Ajayi, Ping Lu, Fisnik Mengjiqi, Roman Schindler
-----------------------------------------------------------------------------------
require 'torch'
require 'image'
require 'nn'
require 'optim'
require("logoRec")


-- Function that constructs the neural network
-- Network architecture according to paper "Deep Learning for Logo Recognition" by  
-- Bianco, Buzzelli, Mazzini and Schettini (2017)
--   out: neural network model
function build_network() 
    local model = nn.Sequential()

    -- convolutional layers
    model:add(nn.SpatialConvolution(3, 32, 5, 5,1,1,2,2))
    model:add(nn.SpatialMaxPooling(2,2,2,2))
    model:add(nn.ReLU())

    model:add(nn.SpatialConvolution(32, 32, 5, 5,1,1,2,2))
    model:add(nn.ReLU())
    model:add(nn.SpatialAveragePooling(2,2,2,2))

    model:add(nn.SpatialConvolution(32, 64, 5, 5,1,1,2,2))
    model:add(nn.ReLU())
    model:add(nn.SpatialAveragePooling(2,2,2,2))

    -- reshape from 3d tensor of 64x4x4 to 1d tensor 64*4*4
    model:add(nn.View(64*4*4))

    -- fully connected layers and softmax
    model:add(nn.Linear(64*4*4,64))
    model:add(nn.Linear(64,33))
    model:add(nn.LogSoftMax())

    return model
end


-- Function that selects a batch of samples from the training set for sgd
-- (partially based on github.com/torch/tutorials/blob/master/2_supervised/4_train.lua)
--   in: dataset (tensor): nSamples x 3x32x32 DoubleTensor containing the training data
--       labels (tensor): Vector containing the true label of each sample
--       size (int): desired batch size
--   out: A batch of samples of the desired size (size x 3x32x32 DoubleTensor)
function get_next_batch(dataset, labels, size) 
    -- get the indices of the samples of the batch
    local indices = {}
    for i = batch_idx,math.min(batch_idx+size-1,labels:size(1)) do
        table.insert(indices,shuffle[i])
    end 

    -- create tensors with the data/labels from the samples with the selected indices
    local batch_data = torch.DoubleTensor(#indices,3,32,32)
    local batch_labels = torch.DoubleTensor(#indices)
    for i = 1, #indices do
        batch_data[i]:copy(dataset[indices[i]])
        batch_labels[i] = labels[indices[i]]
    end
    return batch_data, batch_labels
end


-- Function that trains the network
--   in: train_images_path (string): name of folder with the training images
--       learning_rate (double): the learning rate
--       nEpochs (int): number of epochs
--       batch_size (int): batch size
--   out: The trained model
function train_model(train_images_path, learning_rate, nEpochs, batch_size)
    -- Get number of preprocessed training images
    local nb_files = 0
    for file in paths.files(train_images_path) do
        if file:find('.jpg$') then
            nb_files = nb_files+1
        end
    end

    -- Read training images and their labels and store them in tensors
    local train_data = torch.DoubleTensor(nb_files,3,32,32)
    local train_labels = torch.DoubleTensor(nb_files)
    local idx = 1
    for file in paths.files(train_images_path) do
        if file:find('.jpg$') then
            -- extract and store label of current file
            i,j = string.find(file, "%a+")
            local label = string.sub(file,i,j)
            train_labels[idx] = labelMapping[label]

            -- read and store image data of current file
            local filename = train_images_path .. '/' .. file
            local img = image.load(filename,3,'byte')
            train_data[idx]:copy(img:double())
            idx = idx+1
        end
    end

    -- randomly permute the training samples (so batches contain different classes)
    shuffle = torch.randperm(nb_files)

    -- build model
    local model = build_network()
    local criterion = nn.ClassNLLCriterion()
    local optimState = {learningRate = learning_rate}
    local parameters, gradParameters = model:getParameters()

    feval = function(x)
        local batch_data, batch_labels = get_next_batch(train_data, train_labels, batch_size)
        model:zeroGradParameters()
        local outputs = model:forward(batch_data)
        err = criterion:forward(outputs,batch_labels)
        local gradOutputs = criterion:backward(outputs,batch_labels)
        model:backward(batch_data, gradOutputs)
        return err, gradParameters
    end

    -- run the optimization
    for i = 1, nEpochs do
        batch_idx = 1
        for j = 1, nb_files, batch_size do
            optim.adadelta(feval,parameters,optimState)
            --optim.adam(feval,parameters,optimState)
            --print(err)
            batch_idx = batch_idx + batch_size
        end
        print('epoch: ' .. tostring(i) .. ', error: ' .. tostring(err))
    end
    
    -- return the trained model
    return model
end



