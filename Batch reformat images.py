# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 07:27:07 2018
These functions takes all the image files within a folder and the subfolders,
resizes/reformats them, and saves them into a destination folder. This is useful
if, for example, a standard CNN accepts matrices of a certain size (n x n), but
your images are of a different size. For example, inception v3 accepts 
299 x 299 images, but for many of my projects, the image size is 2000 x 2000, 
1024 x 1024, 512 x 512...etc. It saves time during testing out algorithms
if you don't have to reformat the images every time you restart the kernel.

Also included are some count functions, which help figure out how many images
you actually have labeled.
@author: Xornack
"""
import os
import pydicom
from skimage import transform as sk_transform
import matplotlib.pyplot as plt
import numpy as np

# Functions to parse a path name to isolate images (including dicom if needed).
# I've seen examples using glob, but I always have formatting issues with glob,
# and those make it hard to change later. I think it's simpler to use a custom
# function. At least then I control the formatting. The other, simpler, option
# is to use os.listdir and just make sure there aren't any non-image files in 
# the folder.
def parse_path(path, 
               image_formats = ['png', 'jpg', 'jpeg', 'dcm']):
    form = path.split('.')[-1]
    if form in image_formats:
        return path # returns only the allowed image formats, otherwise None

def list_image_paths(folder_path, # input the folder containing images.
                     image_formats = ['png', 'jpg', 'jpeg', 'dcm']):
    return [folder_path + '/' + i for i in os.listdir(folder_path) 
            if i.split('.')[-1] in image_formats]

# List all subfolder paths if you want to keep all your project images
# in separate folders. Just as a convention, I leave the '/' off the end of
# the path names.
def list_subfolder_image_paths(folder):
    subfolder_paths = [folder + '/' + i for i in os.listdir(folder)]
    files = [list_image_paths(i) for i in subfolder_paths]
    return [i for j in files for i in j]

# Selective list of subfolder paths.
def list_subfolder_image_paths_selective(folder, selective_list = []):
    subfolder_paths = [folder + '/' + i for i in os.listdir(folder) \
                       if i in selective_list]
    files = [list_image_paths(i) for i in subfolder_paths]
    return [i for j in files for i in j]

# Can sometimes to useful to count the number of files in each folder.
def count_files_in_subfolders(folder):
    print("There are ", len(list_subfolder_image_paths(folder)), "total files.")
    subfolder_paths = [folder + '/' + i for i in os.listdir(folder)]
    files = [list_image_paths(i) for i in subfolder_paths]
    counts = [len(list_subfolder_image_paths(folder))]
    for i in range(len(files)):
        folder_name = subfolder_paths[i].split('/')[-1]
        print("There are ", len(files[i]), "files in ", folder_name)
        counts.append(len(files[i]))
        
    return counts # Return a list of the values in case you want to use it.

# Same idea as above. You sometimes want a count of certain folders only.
def count_files_in_subfolders_selective(folder, selective_list = []):
    print("There are ", len(list_subfolder_image_paths(folder)), "total files.")
    subfolder_paths = [folder + '/' + i for i in os.listdir(folder) \
                       if i in selective_list]
    files = [list_image_paths(i) for i in subfolder_paths]
    counts = [len(list_subfolder_image_paths(folder))]
    for i in range(len(files)):
        folder_name = subfolder_paths[i].split('/')[-1]
        print("There are ", len(files[i]), "files in ", folder_name, ".")
        counts.append(len(files[i]))
    print("There are ", sum(counts[1:]), "files in the selected folders.")
        
    return counts

'''
This code takes those path lists from before and does a batch reformat of 
some kind. This is just useful starter code. Projects often require their own 
standardizations, normalizations, and transformations.
'''
# Format file with some input parameters. Saves it to a new folder.
# x and y are the desired dimensions of the output file.
# This code reformats for inception v3.
# Saves the image as whatever format it originally was.
def reformat_and_copy(image_path, 
                      x, # = 299 for inception v3
                      y, 
                      output_path): # complete output paths
    
    # Read in pixel data and file name.
    image = plt.imread(image_path)
    file_name = "reform_incept_v3_" + image_path.split('/')[-1]
        
    # Resize image.
    image = sk_transform.resize(image, (x, y))
    
    # Before I do anything, I need to make sure the image is grayscale and one
    # channel. The processing is just easier, and all my images are grayscale
    # for radiology anyway. Can be done with color images too...for another day.
    # Also, it's much faster to resize the image before you do this operation.
    # Note: the actual inception format requires three channels, but it's easy to
    # read in RGB from grayscale, and it saves a bit of space to have one channel.
    # I could go either way on it though.
    if len(image.shape) > 2:
        temp_row = []
        temp = []
        for i in image:
            for j in i:
                temp_row.append(j[0])
            temp.append(temp_row)
            temp_row = []
        
        image = np.array(temp)
    
    # inception v3 expects the pixel values to be between -1 and 1.
    image = (2*image - np.max(image))/np.max(image)
    
    # Copy the file to disc. If the output_path doesn't exist, it makes one.
    if not os.path.exists(output_path):       
            os.makedirs(output_path)
    plt.imsave(output_path + file_name, image, cmap = 'gray')
    
    return None
    
# Pretty much the same as above, except it works on dicom files.
# Saves the image file as a .png.
def reformat_and_copy_DICOM(file_path, x, y, output_path):
    
    # Read in file and just get the pixel data.
    image = pydicom.read_file(file_path)
    file_name = str(image.PatientName) # or whatever you want to name your file
    image = image.pixel_array

    # Resize image.
    image = sk_transform.resize(image, (x, y))
    
    # inception v3 expect the image to be between -1 and 1.
    image = 2*image - np.max(image)
    
    # Copy the file to disc. If the output_path doesn't exits, it makes one.
    if not os.path.exists(output_path):          
            os.makedirs(output_path)
    plt.imsave(output_path + file_name + ".png", image, cmap = 'gray')
    
    return None

# The most generic function: just copy image to a new location.
def copy_image(image_path, output_path):
    image = plt.imread(image_path)
    file_name = "copied " + image_path.split('/')[-1]    
    
    if not os.path.exists(output_path):          
        os.makedirs(output_path)
    plt.imsave(output_path + file_name, image, cmap = 'gray')

'''
This is an example call using the above functions:

for i in list_subfolder_image_paths_selective("E:/folder", ['label1']):
    reformat_and_copy(i, 299, 299, "E:/reformatted_folder/")
    
I've used these functions to reformat or just sort images in a similar way.
'''














