import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist 
 
#Function to implement steps given in previous section
def kmeans(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    print(idx)
    #Randomly choosing Centroids 
    centroids = x[idx, :] 
     
    #finding the distance between centroids and all the data points
    distances = cdist(x, centroids ,'euclidean') 
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) 
    
    # just to ensure that there is no problem due to any 
    # 2 random centroids accidently being equal
    for i in range(k):
        points[idx[i]] = i
        
    
    #Repeating the above steps for a defined number of iterations
    for itr in range(no_of_iterations): 
        centroids = []
        print(f"*** Iteration {itr} ***")
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            print(x[points==idx])
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids 
         
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
         
    return centroids

""" 
pads the image with itself, if image is not completely
dividable in patches of patch_shape
"""
def padImageForPatches(img, patch_shape):
    return cv2.copyMakeBorder(src=img, left=0, right=patch_shape[1]*((img.shape[1]+31)//patch_shape[1])-img.shape[1], top=0, bottom=patch_shape[0]*((img.shape[0]+31)//patch_shape[0])-img.shape[0], borderType=cv2.BORDER_REFLECT)

def divideIntoPatches(img, patch_shape):
    img = padImageForPatches(img, patch_shape)
    img_shape = img.shape
    patch_array = np.empty((int((img_shape[0]*img_shape[1])/(patch_shape[0]*patch_shape[1])), patch_shape[0], patch_shape[1], 3), dtype=np.int32)
    for i in range(0, int(img_shape[0]/patch_shape[0])):
        for j in range(0, int(img_shape[1]/patch_shape[1])):
            patch_array[i*int(img_shape[1]/patch_shape[1]) + j] = img[i*patch_shape[0]:i*patch_shape[0]+patch_shape[0], j*patch_shape[1]:j*patch_shape[1]+patch_shape[1], :]

    return patch_array

def getColorHistFeatures(img, n_bins=8):
    img = img//int(256/n_bins)
    hist = np.bincount(img[:, :, 0].ravel(), minlength=8)
    hist = np.append(hist, np.bincount(img[:, :, 1].ravel(), minlength=8))
    hist = np.append(hist, np.bincount(img[:, :, 2].ravel(), minlength=8))
    return hist


def getHistForPatches(img, patch_shape):
    patch_array = divideIntoPatches(img, patch_shape)
    n, h, w, c = patch_array.shape
    hist_array = np.empty((n, 24), dtype=np.int32)
    for i in range(0, n):
        hist_array[i] = getColorHistFeatures(patch_array[i])
    return hist_array

def loadImagesFromDir(dir_path):
    images = {}
    
    for folder in os.listdir(dir_path):
        img_arr = []
        folder_path = dir_path + "/" + folder
        for file in os.listdir(folder_path):
            img_path = folder_path + "/" + file
            img = cv2.imread(img_path)
            if img is not None:
                img_arr.append(img)
        images[folder] = img_arr
    
    return images

def getHistForAllImages(images, patch_shape):
    images_hist_features = {}

    for key, img_arr in images.items():
        hist24_arr = []
        for img in img_arr:
            hist24_arr.append(getHistForPatches(img, patch_shape))
        images_hist_features[key] = hist24_arr
    
    return images_hist_features

def BoVW_image_feature_vector(img_col_hist, centroids):
    distances = cdist(img_col_hist, centroids ,'euclidean')
    points = np.array([np.argmin(i) for i in distances])
    feature = np.bincount(points, minlength=len(centroids))
    return feature

train_images = loadImagesFromDir("Group21/Classification/Image_Group21/train")
test_images = loadImagesFromDir("Group21/Classification/Image_Group21/test")


train_imgs_hist = getHistForAllImages(train_images, patch_shape=(32, 32))
test_imgs_hist = getHistForAllImages(test_images, patch_shape=(32, 32))

train_imgs_hist_flattened = np.concatenate([y for x in train_imgs_hist.values() for y in x])

# print(train_imgs_hist_flattened)

centroids = kmeans(train_imgs_hist_flattened, 32, 100)

img_BoVW_all = {}
for img_type, img_hist_array in train_imgs_hist.items():
    img_BoVW = np.empty((len(img_hist_array), 32), dtype=np.int32)
    for i in range(len(img_hist_array)):
        img_BoVW[i] = BoVW_image_feature_vector(img_hist_array[i], centroids)
    img_BoVW_all[img_type] = img_BoVW

img_BoVW_all["batters_box"][0]