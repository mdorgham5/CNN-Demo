# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:56:10 2017

@author: dorgham
"""

import numpy as np
import scipy.misc
from scipy import ndimage
import Image
import glob

#URL for the dataset
#http://www2.informatik.uni-freiburg.de/~spinello/RGBD-dataset.html 
txtFiles = glob.glob("/home/dorgham/Desktop/IAS/ComputerVisionProject/datasets/mensa_seq0_1.1/track_annotations/*.txt")

for txtfile in txtFiles:
    file = open(txtfile, 'r')
    for line in file:
        if not line.startswith("#"):
            tokens = line.split( )
            xTopleft = int(tokens[6])
            if xTopleft<0:
                xTopleft=0
            yTopleft = int(tokens[7])
            if yTopleft<0:
                yTopleft=0
            rgbWidth = int(tokens[8])
            rgbHeight = int(tokens[9])
            imgFileName = ("/home/dorgham/Desktop/IAS/ComputerVisionProject/datasets/"
            "mensa_seq0_1.1/rgb/" + tokens[0] + ".ppm")
            image = Image.open(imgFileName)
            # Converting an Image into a numpy array for computations
            image_array = np.array(image)
            #print image_array.shape
            print txtfile, tokens[0]
            print yTopleft, yTopleft+rgbHeight, xTopleft, xTopleft+rgbWidth
            croppedImage = image_array[yTopleft:yTopleft+rgbHeight, xTopleft:xTopleft+rgbWidth]
            #yTopleft:rgbHeight, xTopleft:rgbWidth
            outputFileName = ("/home/dorgham/Desktop/IAS/ComputerVisionProject/datasets/"
            "mensa_seq0_1.1/cropped/"+tokens[0]+tokens[1]+tokens[2]+tokens[6]+tokens[7]+
            tokens[8]+tokens[9]+".ppm")
            rotatedImage = ndimage.rotate(croppedImage, 90)
            size = 360, 180
            resizedIamge = scipy.misc.imresize(rotatedImage, size)
            scipy.misc.imsave(outputFileName, resizedIamge)
    
    
