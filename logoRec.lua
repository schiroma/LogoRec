-----------------------------------------------------------------------------------
-- Implementation of a RCNN for logo recognition for the 'Advanced Methods in
-- Machine Learning' course (SS2017)
--
-- Authors: Tofunmi Ajayi, Ping Lu, Fisnik Mengjiqi, Roman Schindler
-----------------------------------------------------------------------------------
require 'torch'
require 'image'
require 'nn'
require 'optim'
require 'nngraph'


-- Global variables
dataset_path = "FlickrLogos-v2/"
bbox_path = "FlickrLogos-v2/classes/masks/"
images_path = "FlickrLogos-v2/classes/jpg/"


-- Define the label mappings (In the network, each class is represented by a number)
labelMapping = {}
labelMapping['adidas'] = 1
labelMapping['aldi'] = 2
labelMapping['apple'] = 3
labelMapping['becks'] = 4
labelMapping['bmw'] = 5
labelMapping['carlsberg'] = 6
labelMapping['chimay'] = 7
labelMapping['cocacola'] = 8
labelMapping['corona'] = 9
labelMapping['dhl'] = 10
labelMapping['erdinger'] = 11
labelMapping['esso'] = 12
labelMapping['fedex'] = 13
labelMapping['ferrari'] = 14
labelMapping['ford'] = 15
labelMapping['fosters'] = 16
labelMapping['google'] = 17
labelMapping['guiness'] = 18
labelMapping['heineken'] = 19
labelMapping['HP'] = 20
labelMapping['milka'] = 21
labelMapping['nvidia'] = 22
labelMapping['paulaner'] = 23
labelMapping['pepsi'] = 24
labelMapping['rittersport'] = 25
labelMapping['shell'] = 26
labelMapping['singha'] = 27
labelMapping['starbucks'] = 28
labelMapping['stellaartois'] = 29
labelMapping['texaco'] = 30
labelMapping['tsingtao'] = 31
labelMapping['ups'] = 32
labelMapping['no-logo'] = 33
labelMapping[1] = 'adidas'
labelMapping[2] = 'aldi'
labelMapping[3] = 'apple'
labelMapping[4] = 'becks'
labelMapping[5] = 'bmw'
labelMapping[6] = 'carlsberg'
labelMapping[7] = 'chimay'
labelMapping[8] = 'cocacola'
labelMapping[9] = 'corona'
labelMapping[10] = 'dhl'
labelMapping[11] = 'erdinger'
labelMapping[12] = 'esso'
labelMapping[13] = 'fedex'
labelMapping[14] = 'ferrari'
labelMapping[15] = 'ford'
labelMapping[16] = 'fosters'
labelMapping[17] = 'google'
labelMapping[18] = 'guiness'
labelMapping[19] = 'heineken'
labelMapping[20] = 'HP'
labelMapping[21] = 'milka'
labelMapping[22] = 'nvidia'
labelMapping[23] = 'paulaner'
labelMapping[24] = 'pepsi'
labelMapping[25] = 'rittersport'
labelMapping[26] = 'shell'
labelMapping[27] = 'singha'
labelMapping[28] = 'starbucks'
labelMapping[29] = 'stellaartois'
labelMapping[30] = 'texaco'
labelMapping[31] = 'tsingtao'
labelMapping[32] = 'ups'
labelMapping[33] = 'no-logo'


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
        file:close()
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
        file:close()
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
    local x1 = bbox[{1,1}]
    local y1 = bbox[{1,2}]
    local x2 = x1 + bbox[{1,3}] - 1
    local y2 = y1 + bbox[{1,4}] - 1
    return image.crop(img,x1,y1,x2,y2)
    --return img:sub(1,3,y1+1,y2+1,x1+1,x2+1)
end


-- Function that generates the actual images used for training (cut out logos and
-- region proposals from the training images)
-- Note: Will eventually be removed since we do the preprocessing with python
--   Arguments: trainset (string) - ...
--   Returns: table containing the generated images we use for training
function generate_training_data(trainset)
    samples = read_data(trainset)

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

    -- scale all generated samples to 3x32x32
    for i,sample in ipairs(training_data) do
        sample.img = image.scale(sample.img,32,32)
    end

    -- Save the generated images to folder
    for i,sample in ipairs(training_data) do
        save_image(sample.label .. '_' .. tostring(i) .. '.jpg', sample.img)
    end

    -- return the generated training samples
    return training_data
end


-- Function that saves an image as jpg file
--   Arguments: filename - name of the file (e.g. 'test.jpg')
--              img (ByteTensor) - 3xmxn ByteTensor representing an image
function save_image(filename, img)
    image.save("output/" .. filename,img)
end

