-----------------------------------------------------------------------------------
-- Implementation of a RCNN for logo recognition for the 'Advanced Methods in
-- Machine Learning' course (SS2017)
--
-- Authors: Tofunmi Ajayi, Ping Lu, Fisnik Mengjiqi, Roman Schindler
-----------------------------------------------------------------------------------
require 'torch'
require 'image'


-- Global variables
dataset_path = "FlickrLogos-v2/"
bbox_path = "FlickrLogos-v2/classes/masks/"
images_path = "FlickrLogos-v2/classes/jpg/"


-- Function to read data files (containing image-filenames and contained logo)
--   Arguments: data_file (string) - filename of the data file
--   Returns: table containing the samples which contain label, logo bounding box 
--            and filename of the corresponding image
function read_data(data_file)
    -- open the specified txt-file and read it line by line
    local samples = {}
    local file = io.open(dataset_path .. data_file)
    if file then
        for line in file:lines() do
            -- read label and filename
            local label, filename = unpack(line:split(","))
            filename = string.sub(filename,1,#filename-1)

            -- from the filename, construct filename of corresponding bbox-file
            local bbox_filename = bbox_path .. label .. "/" .. filename .. ".bboxes.txt"

            -- create sample i.e. a table containing label, filename and bbox
            local sample = {}
            sample.label = label
            sample.image_file = filename
            sample.bbox = read_boundingboxes(bbox_filename)

            -- add new sample to the collection of samples
            table.insert(samples,sample)
        end
    end

    return samples
end


-- Function to read bounding box files which contain the bounding box of the logo in an image
--   Arguments: bbox_file (string) - filename of the bounding box file
--   Returns: 1x4 tensor (x/y of upper left corner, width and height)
function read_boundingboxes(bbox_file)
    local boundingbox = torch.Tensor(1,4)
    local file = io.open(bbox_file)
    if file then
        local header = true
        for line in file:lines() do
            if not header then
                local x, y, w, h = unpack(line:split(" "))
                boundingbox[{1,1}] = tonumber(x)
                boundingbox[{1,2}] = tonumber(y)
                boundingbox[{1,3}] = tonumber(w)
                boundingbox[{1,4}] = tonumber(h)
            else
                header = false
            end
        end
    end
    return boundingbox
end


-- Function that cuts out the logo from an image using the bounding box
--   Arguments: sample (table) - a sample-table containing image-file, label and bbox
--   Returns: 3xmxn ByteTensor representing an image of the logo
function extract_logo(sample)
    -- load the image using the samples image-filename  
    local img_filename = images_path .. sample.label .. "/" .. sample.image_file
    local img = image.load(img_filename,3,'byte')

    -- extract the logo-part from the image-matrix using the bounding box
    local logo = crop_image(img,sample.bbox)
    return logo
end


-- Function that crops an image given a bounding box
--   Arguments: img (ByteTensor) - 3xmxn matrix representing an image
--              bbox (tensor) - 1x4 vector (x/y of upper left corner, width and height)
--   Returns: 3xmxn ByteTensor representing the cropped image
function crop_image(img, bbox)
    local x1 = bbox[{1,1}] - 1
    local y1 = bbox[{1,2}] - 1
    local x2 = x1 + bbox[{1,3}] - 2
    local y2 = y1 + bbox[{1,4}] - 2
    return image.crop(img,x1,y1,x2,y2)
    --return img:sub(1,3,y1+1,y2+1,x1+1,x2+1)
end


-- Function that generates annotated region proposals from an image using selective search
--   Arguments: sample (table) - a sample containing image-file, label and bbox
--   Returns: table containing image-regions and their label
--function generate_region_proposals(sample)
    -- TODO
    -- load image
    -- img = ...
    --
    -- generate region proposals
    -- ... = selective_search(img)
    --
    -- annotate each region with a label
    -- for each region
    --     local generated_sample = {}
    --     local label = annotate_region(sample,region)
    --     local region = crop_image(img,region)
    --     generated_sample.img = region
    --     generated_sample.label = label
    --     table.insert(regions,generated_sample)
    --
    -- return regions
--end


-- Function that generates region proposals from an image using selective search
-- (too hard to implement. we should try to call an existing c++ or python implementation)
--   Arguments: img (ByteTensor) - 3xmxn matrix representing an image
--   Returns: table of bounding boxes representing the regions
--function selective_search(img)
    -- TODO
--end


-- Function that annotates an image region with a label (the logo name or 'no logo')
-- by comparing it with the ground truth bounding box of the logo 
--   Arguments: sample (table) - a sample-table containing image-file, label and bbox
--              region (tensor) - 1x4 vector (x/y of upper left corner, width and height)
--   Returns: string representing the label of the input region
--function annotate_region(sample, region)
    -- TODO
    -- compute IoU (intersection over union) between logo-bbox and region
    -- ...
    --
    -- label = 'nologo'
    -- if (IoU > 0.5) then
    --     label = sample.label
    -- end
    -- return label  
--end


-- Function that generates the actual images used for training (cut out logos and
-- region proposals from the training images)
--   Arguments: samples (table) - a table containing all the loaded training samples
--   Returns: table containing the generated images we use for training
function generate_training_data(samples)
    -- table that stores the generated training images
    local training_data = {}

    -- extract all logos from the training images
    for i,sample in ipairs(samples) do
        local generated_sample = {}
        local logo = extract_logo(sample)
        generated_sample.img = logo
        generated_sample.label = sample.label
        table.insert(training_data,generated_sample)
    end

    -- generate region proposals using selective search
    -- TODO  
    --for i,sample in ipairs(samples) do
        --local generated_samples = generate_region_proposals(sample)
        --table.insert(training_data,generated_samples)
    --end  

    -- scale all generated samples to 3x32x32
    for i,sample in ipairs(training_data) do
        sample.img = image.scale(sample.img,32,32)
    end

    -- return the generated training samples
    return training_data
end


-- Function that constructs the neural network
-- Network architecture according to paper "Deep Learning for Logo Recognition" by Bianco, 
-- Buzzelli, Mazzini and Schettini (2017)
--   Returns: neural network model
function build_network() 
    local model = nn.Sequential()

    -- convolutional layers
    model.add(nn.SpatialConvolution(3, 32, 5, 5,1,1,2,2))
    model.add(nn.SpatialMaxPooling(2,2,2,2))
    model.add(nn.ReLU())

    model.add(nn.SpatialConvolution(32, 32, 5, 5,1,1,2,2))
    model.add(nn.ReLU())
    model.add(nn.SpatialAveragePooling(2,2,2,2))

    model.add(nn.SpatialConvolution(32, 64, 5, 5,1,1,2,2))
    model.add(nn.ReLU())
    model.add(nn.SpatialAveragePooling(2,2,2,2))

    -- reshape from 3d tensor of 64x4x4 to 1d tensor 64*4*4
    model.add(nn.View(64*4*4))

    -- fully connected layers and softmax
    model.add(nn.Linear(64*4*4,64))
    model.add(nn.Linear(64,33))
    model:add(nn.LogSoftMax())

    return model
end


-- Function that saves an image as jpg file
--   Arguments: filename - name of the file (e.g. 'test.jpg')
--              img (ByteTensor) - 3xmxn ByteTensor representing an image
function save_image(filename, img)
    image.save("output/" .. filename,img)
end

------------------------------------------------------------------------------
-- Read the training images
train_samples = read_data("trainset.txt")

-- Generate the training data from the training images
training_data = generate_training_data(train_samples)

-- Save the generated images to folder
for i,sample in ipairs(training_data) do
    save_image('tr' .. tostring(i) .. '.jpg', sample.img)
end


