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
require 'gnuplot'
--require 'cutorch'
--require 'cunn'


-- Global variables
dataset_path = "FlickrLogos-v2/"
bbox_path = "FlickrLogos-v2/classes/masks/"
images_path = "FlickrLogos-v2/classes/jpg/"
roi_path = "regions/"

-- The different classes
classes ={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}

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
labelMapping['no'] = 33
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
--   in: data_file (string) - filename of the data file
--       positives_only (bool) - if true, only samples with a logo are loaded
--   out: table containing the samples which contain label, logo bounding box 
--        and filename of the corresponding image
function read_data(data_file, positives_only)
    -- open the specified txt-file and read it line by line
    local samples = {}
    local file = io.open(dataset_path .. data_file)
    if file then
        for line in file:lines() do
            -- read label and filename
            local label, filename = unpack(line:split(","))
            filename = string.sub(filename,1,#filename-1)

            -- skip no-logo samples if we want positive samples only
            if ((label ~= 'no-logo') or (positives_only == false)) then

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
        file:close()
    end
    return samples
end


-- Function to read bounding box files which contain the bounding box of the logo in an image
--   in: bbox_file (string) - filename of the bounding box file
--   out: table of bboxes (each bbox a table with x and y of upper left corner, width and height)
function read_boundingboxes(bbox_file)
    local boundingboxes = {}
    local file = io.open(bbox_file)
    if file then
        local header = true
        for line in file:lines() do
            if not header then
                local x, y, w, h = unpack(line:split(" "))
                local boundingbox = {}
                boundingbox['x'] = tonumber(x)
                boundingbox['y'] = tonumber(y)
                boundingbox['w'] = tonumber(w)
                boundingbox['h'] = tonumber(h)
                table.insert(boundingboxes,boundingbox)
            else
                header = false
            end
        end
        file:close()
    else
        -- dummy bounding box for images with no logos
        local boundingbox = {}
        boundingbox['x'] = 1
        boundingbox['y'] = 1
        boundingbox['w'] = 0
        boundingbox['h'] = 0
        table.insert(boundingboxes,boundingbox)
    end
    return boundingboxes
end


-- Function that searches regions of interest by selective search. The selective
-- search results for each image were precomputed with a python implementation
-- and stored to txt-files. This function just reads these precomputed results
-- and is thus just imitating an actual implementation of selective search
--   in: sample (table) - a sample-table containing image-file, label and bbox
--   out: table containing the regions (x,y,w,h-table) for the input image
function selective_search(sample)
    local regions = {}
    local filename = string.sub(sample.image_file,1,#sample.image_file-4)
    filename = roi_path .. filename .. ".txt"
    local file = io.open(filename)
    if file then
        local header = true
        for line in file:lines() do
            if not header then
                local x, y, w, h = unpack(line:split(" "))
                local region = {}
                region['x'] = tonumber(x)
                region['y'] = tonumber(y)
                region['w'] = tonumber(w)
                region['h'] = tonumber(h)
                table.insert(regions,region)
            else
                header = false
            end
        end
        file:close()
    end
    return regions
end


-- Fuction that annotates a region with a label by comparing the region
-- to the ground truth logo bounding box
--   in: sample (table) - a sample-table containing image-file, label and bbox
--       region (table) - a table containing x,y,w,h of the region to annotate
--   out: The label of the region (either a logo-name or 'no-logo')
function annotate_region(sample, region) 
    local label = sample.label

    -- compare region to every logo-boundingbox of the sample
    for i,bbox in ipairs(sample.bbox) do
        local iou = intersection_over_union(bbox, region)
        if (iou > 0.4) then
            return label
        end
    end
    return 'no-logo'
end


-- Function to compute the IoU between two rectangles
--   in: bbox (table) - a table containing x,y,w,h describing a rectangle
--       region (table) - a table containing x,y,w,h describing a rectangle
--   out: The IoU between the two rectangles (a value between 0 and 1)
function intersection_over_union(bbox,region)
    -- get coordinates of two corners of first rectangle
    local a_x1 = bbox['x']
    local a_y1 = bbox['y']
    local a_x2 = a_x1 + bbox['w']
    local a_y2 = a_y1 + bbox['h']

    -- get coordinates of two corners of second rectangle
    local b_x1 = region['x']
    local b_y1 = region['y']
    local b_x2 = b_x1 + region['w']
    local b_y2 = b_y1 + region['h']

    -- return 0 if no intersection at all
    if (a_x1>b_x2 or a_x2<b_x1 or a_y1>b_y2 or a_y2<b_y1) then
        return 0.0
    end
	
    -- determine the (x, y)-coordinates of the intersection rectangle
    local xA = math.max(a_x1, b_x1)
    local yA = math.max(a_y1, b_y1)
    local xB = math.min(a_x2, b_x2)
    local yB = math.min(a_y2, b_y2)

    -- compute the area of intersection rectangle
    local interArea = (xB - xA + 1) * (yB - yA + 1)
	
    -- compute the area of both the prediction and ground-truth rectangles
    local boxAArea = (a_x2 - a_x1 + 1) * (a_y2 - a_y1 + 1)
    local boxBArea = (b_x2 - b_x1 + 1) * (b_y2 - b_y1 + 1)
	
    -- compute the intersection over union by taking the intersection
    -- area and dividing it by the sum of prediction + ground-truth
    -- areas - the interesection area (+0.001 in case of division by zero)  
    local iou = interArea / (boxAArea + boxBArea - interArea + 0.001)
    return iou   
end


-- Function to blur an image with a gaussian convolution
--   in: img (tensor) - a bytetensor representing the image to blur
--   out: the blurred image (by 20x20 kernel) as bytetensor
function blur_image(img)
    local kernel = image.gaussian(20,1,0.25,true)
    img = img:double()/255
    img = image.convolve(img,kernel,'same')
    img = img*255
    img = img:byte()
    return img
end


-- Function to load an image from a sample-table
--   in: sample (table) - a sample-table containing image-file, label and bbox
--   out: a bytetensor representing the loaded image
function load_image(sample)
    -- load the image using the sample's image-filename  
    local img_filename = images_path .. sample.label .. "/" .. sample.image_file
    local img = image.load(img_filename,3,'byte')
    return img
end


-- Function that cuts out a specified area from an image
--   in: img (tensor) - a bytetensor representing the image to blur
--       region (table) - a table containing x,y,w,h of the desired region to crop
--   out: 3xmxn ByteTensor representing the cropped region
function crop_region(img, region)
    -- extract the specified region from the image-matrix 
    local x1 = region['x']
    local y1 = region['y']
    local x2 = x1 + region['w']
    local y2 = y1 + region['h']
    local cropped_region = image.crop(img,x1,y1,x2,y2) 
    return cropped_region
end


-- Function that generates the actual images used for training (cut out logos and
-- region proposals from the training images)
--   in: output_folder (string) - name of the folder to store the data
--       trainset (string) - name of the txt-file containing the image filenames 
--                           and their logo
--       augmentation (bool) - if true, we add blurred images to the dataset
--   out: table containing the generated images we use for training
function generate_training_data(output_folder, trainset, augmentation)
    -- load the specified dataset and only samples that contain a logo
    samples = read_data(trainset,true)

    -- table that stores the generated training images
    local training_data = {}

    -- extract all logos from the training images using gt-bounding box
    for i,sample in ipairs(samples) do
        local img = load_image(sample)
        if augmentation then
            img_blurred = blur_image(img)
        end
        for j,bbox in ipairs(sample.bbox) do
            local generated_sample = {}
            local logo = crop_region(img,bbox)
            generated_sample.img = image.scale(logo,32,32)
            generated_sample.label = sample.label
            table.insert(training_data,generated_sample)
            if (augmentation) then
                local generated_sample_blurred = {}
                logo_blurred = crop_region(img_blurred,bbox)
                generated_sample_blurred.img = image.scale(logo_blurred,32,32)
                generated_sample_blurred.label = sample.label
                table.insert(training_data,generated_sample_blurred)
            end
        end
    end

    -- compute region proposals for each training image and annotate them
    for i,sample in ipairs(samples) do
        local regions = selective_search(sample)
        local img = load_image(sample)
        if augmentation then
            img_blurred = blur_image(img)
        end
        for j,reg in ipairs(regions) do
            local generated_sample = {}
            local label = annotate_region(sample,reg)
            local reg_img = crop_region(img,reg)
            generated_sample.img = image.scale(reg_img,32,32)
            generated_sample.label = label
            table.insert(training_data,generated_sample)  
            if (augmentation) then
                local generated_sample_blurred = {}
                reg_img_blurred = crop_region(img_blurred,reg)
                generated_sample_blurred.img = image.scale(reg_img_blurred,32,32)
                generated_sample_blurred.label = label
                table.insert(training_data,generated_sample_blurred)
            end       
        end
        print('progress: ' .. tostring(i) .. ' / ' .. tostring(#samples))
    end

    -- Save the generated images to folder
    for i,sample in ipairs(training_data) do
        image.save(output_folder .. '/' .. sample.label .. '_' .. tostring(i) .. '.jpg', sample.img)
    end

    -- return the generated training samples
    return training_data
end


