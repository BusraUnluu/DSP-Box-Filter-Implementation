#Busra_Unlu_211711008_HW2

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img_filename = 'lena_grayscale_hq.jpg'
img = cv.imread(img_filename, 0)
(h, w) = img.shape[:2]

# Convolution operation
def convolution(image, filter):                                      #get zero padded img as an input 

    imageW, imageH = image.shape                                     # get image dimensions
    filterW, filterH = filter.shape                                  # get filter dimensions
    concolvedImg = np.zeros((imageW-filterW+1, imageH-filterH+1))    # intialize output image
    #convolve the filter and zero padded image
    for i in range(imageW-filterW+1):
        for j in range(imageH-filterH+1):
            #concolvedImg[i, j] = np.round(np.sum(filter * image[i:i + filterW, j:j + filterH])).astype(np.uint8)
            concolvedImg[i,j] = (np.multiply(filter , image[i:i + filterW, j:j + filterH])).sum()

    
    return concolvedImg


# Box Filter Operation
def boxFilter(image, kernelSize):

    kernel = np.ones((kernelSize,kernelSize))     
    # calculate the kernel average; returns the kernel matrix of the average
    kernel = kernel / (kernelSize ** 2) 
    
    #pad width (P=(F-1)/2) 
    top  = bottom = int((kernelSize-1)/2)        #rows
    left = right  = int((kernelSize-1)/2)        #cols
    padImg = cv.copyMakeBorder( image, top, bottom, left, right, cv.BORDER_CONSTANT, value = 0 )

    #call convolution function with rounding
    filtImg = np.ceil(convolution(padImg, kernel) - 0.5)

    # output image
    filtImg=filtImg.astype(np.uint8)

    # returns the filtered image
    return filtImg


#Box Filter
output_1_1=boxFilter(img, 3)    # kernal size of 3
output_1_2=boxFilter(img, 11)   # kernal size of 11
output_1_3=boxFilter(img, 21)   # kernal size of 21

# opencv box filtering
output_2_1 = cv.boxFilter(img, 0, (3, 3)  , False, borderType=cv.BORDER_CONSTANT)
output_2_2 = cv.boxFilter(img, 0, (11, 11), False, borderType=cv.BORDER_CONSTANT)
output_2_3 = cv.boxFilter(img, 0, (21, 21), False, borderType=cv.BORDER_CONSTANT)

#Question 2
def seperable_boxFilter(image, kernelSize):

    kernel_x = np.ones((1,kernelSize))     
    kernel_y = np.ones((kernelSize,1))
    # calculate the kernel average; returns the kernel matrix of the average
    kernel_x = kernel_x / (kernelSize) 
    kernel_y = kernel_y / (kernelSize)      
    
    #zero pad width (P=(F-1)/2)
    top  = bottom = int((kernelSize-1)/2)        #rows
    left = right  = int((kernelSize-1)/2)        #cols
    
    #0 padded image
    dst = cv.copyMakeBorder( image, top, bottom, left, right, cv.BORDER_CONSTANT, None, value = 0 )

    # Convolve kernel_x with image
    filtImg_x = convolution(dst, kernel_x)

    #rounded concolution output
    filtImg_y = np.ceil(convolution(filtImg_x,kernel_y) - 0.5)

    # output image
    filtImg_y=filtImg_y.astype(np.uint8)

    # returns the filtered image
    return filtImg_y


output_3_1 = seperable_boxFilter(img,3)
output_3_2 = seperable_boxFilter(img,11)
output_3_3 = seperable_boxFilter(img,21)


#outputs
#1 Box filter
cv.imshow('Original image', img)
cv.imshow('Box filtered image (kernel size=3)'      , output_1_1)
cv.imshow('Box filtered image (kernel size=11)'     , output_1_2)
cv.imshow('Box filtered image (kernel size=21)'     , output_1_3)
#2 Opencv Box Filter
cv.imshow('opencv box filtering (kernel size=3)'    , output_2_1)
cv.imshow('opencv box filtering (kernel size=11)'   , output_2_2)
cv.imshow('opencv box filtering (kernel size=21)'   , output_2_3)
#3 Separable Box Filter
cv.imshow('separable box filtering (kernel size=3)' , output_3_1)
cv.imshow('separable box filtering (kernel size=11)', output_3_2)
cv.imshow('separable box filtering (kernel size=21)', output_3_3)
#4- Differences
cv.imshow('Difference_1 (output_1_1 - output_2_1)' , 100*(abs(output_1_1 - output_2_1)))
cv.imshow('Difference_1 (output_1_2 - output_2_2)' , 100*(abs(output_1_2 - output_2_2)))
cv.imshow('Difference_1 (output_1_3 - output_2_3)' , 100*(abs(output_1_3 - output_2_3)))

cv.imshow('Difference_1 (output_3_1 - output_2_1)' , 100*(abs(output_3_1 - output_2_1)))
cv.imshow('Difference_1 (output_3_2 - output_2_2)' , 100*(abs(output_3_2 - output_2_2)))
cv.imshow('Difference_1 (output_3_3 - output_2_3)' , 100*(abs(output_3_3 - output_2_3)))

#TERMINAL OUTPUTS
print("------------------------------------------")
print("Difference (output_1_1 - output_2_1) is : ",    np.sum(np.sum(np.abs(output_1_1 - output_2_1))))
print("------------------------------------------")
print("Difference (output_1_2 - output_2_2) is : ",    np.sum(np.sum(np.abs(output_1_2 - output_2_2))))
print("------------------------------------------")
print("Difference (output_1_3 - output_2_3) is : ",    np.sum(np.sum(np.abs(output_1_3 - output_2_3))))
print("------------------------------------------")
print("Difference (output_2_1 - output_3_1) is : ",    np.sum(np.sum(np.abs(output_3_1 - output_2_1))))
print("------------------------------------------")
print("Difference (output_2_2 - output_3_2) is : ",    np.sum(np.sum(np.abs(output_3_2 - output_2_2))))
print("------------------------------------------")
print("Difference (output_2_3 - output_3_3) is : ",    np.sum(np.sum(np.abs(output_3_3 - output_2_3))))
print("------------------------------------------")

#finding max difference
a=output_1_1 - output_2_1
b=output_1_2 - output_2_2
c=output_1_3 - output_2_3
d=output_3_1 - output_2_1
e=output_3_2 - output_2_2
f=output_3_3 - output_2_3
differences=[a,b,c,d,e,f]
print("Max difference",np.max(differences))

cv.waitKey(0)
cv.destroyAllWindows()