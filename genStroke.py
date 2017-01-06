import cv2
import numpy as np
from matplotlib import pyplot as plt

import math
import sys
import os


def rotate_img(img, angle):
    row, col = img.shape
    M   = cv2.getRotationMatrix2D((row / 2 + 1, col / 2 + 1), angle, 1)
    res = cv2.warpAffine(img, M, (row, col))
    return res

def rotateImg(img, angle):
    row, col = img.shape
    M   = cv2.getRotationMatrix2D((row / 2 + 1, col / 2 + 1), angle, 1)
    res = cv2.warpAffine(img, M, (row, col))
    return res

def genDirectionLines(halfkernel, verbose = False):
    len = halfkernel * 2 + 1
    kernel = np.zeros((8, len, len))
    kernel [0, halfkernel, :] = 1.0
    kernel [2, :, :] = np.rot90(kernel[0,:,:]) 
    kernel [4, :, :] = np.rot90(kernel[2,:,:]) 
    kernel [6, :, :] = np.rot90(kernel[4,:,:]) 
    for i in range(len):
        kernel[1,i,i] = 1.0
    kernel [3, :, :] = np.rot90(kernel[1,:,:]) 
    kernel [5, :, :] = np.rot90(kernel[3,:,:]) 
    kernel [7, :, :] = np.rot90(kernel[5,:,:]) 
    if verbose == True:
        for i in range(8):
            title = 'kernel line %d'%i
            cv2.imshow(title, kernel[i,:,:])
            cv2.waitKey(1)
    return kernel;        
    

# compute and get the stroke of the raw img
def genStroke(img, dirNum, verbose = False):
    height , width = img.shape[0], img.shape[1]
    img = np.float32(img) / 255.0
    print "- Images, size %dx%d"%(width, height)
    print "- PreProcessing Images, denoising ..."
    img = cv2.medianBlur(img, 3)
    if verbose == True:
        cv2.imshow('blurred image', img)
        cv2.waitKey(1)

    print "- Generating Gradient Images ..."
    imX = np.append(np.absolute(img[:, 0 : width - 1] - img[:, 1 : width]), np.zeros((height, 1)), axis = 1)
    imY = np.append(np.absolute(img[0 : height - 1, :] - img[1 : height, :]), np.zeros((1, width)), axis = 0)
    #img_gredient = np.sqrt((imX ** 2 + imY ** 2))
    img_gredient = imX + imY
    if verbose == True:
        cv2.imshow('gradient image', img_gredient)
        cv2.waitKey(1)

    #filter kernel size
    tempsize = 0 
    if height > width:
        tempsize = width
    else:
        tempsize = height
    tempsize /= 32
    halfKsize = tempsize / 2
    if halfKsize < 1:
        halfKsize = 3
    if halfKsize > 9:
        halfKsize = 9
    len = halfKsize * 2 + 1
    print "- Kernel Line Size=%s" %(len)
    kernel = np.zeros((dirNum, len, len))
    kernel [0,halfKsize,:] = 1.0
    #kernel = genDirectionLines(halfKsize, False);
    for i in range(1,dirNum):
        kernel[i,:,:] = temp = rotateImg(kernel[0,:,:], i * 180 / dirNum)
        if verbose == True:
            title = 'line kernel %d'%i
            cv2.imshow( title, temp)
            cv2.waitKey(1)

    #filter gradient map in different directions
    print "- Filtering Gradient Images in different directions ..."
    response = np.zeros((dirNum, height, width))
    for i in range(dirNum):
        ker = kernel[i,:,:]; 
        response[i, :, :] = cv2.filter2D(img_gredient, -1, ker)
    if verbose == True:
        for i in range(dirNum):
            title = 'response %d'%i
            cv2.imshow( title, response[i,:,:])
            cv2.waitKey(1)

    #divide gradient map into 8 different sub-map
    print "- Caculating Gradient classification ..."
    Cs = np.zeros((dirNum, height, width))
    for x in range(width):
        for y in range(height):
            i = np.argmax(response[:,y,x])
            Cs[i, y, x] = img_gredient[y,x]

    #generate line shape
    print "- Generating shape Lines ..."
    spn = np.zeros((dirNum, height, width))
    for i in range(dirNum):
        ker = kernel[i,:,:]; 
        spn[i, :, :] = cv2.filter2D(Cs[i], -1, ker)
    sp = np.sum(spn, axis = 0)
    sp =  (sp - np.min(sp)) / (np.max(sp) - np.min(sp))
    S  = 1 -  sp
    if verbose == True:
        cv2.imshow('raw stroke', sp)
        cv2.waitKey(1)
        cv2.imshow('stroke', S)
        cv2.waitKey(1)

    return S

if __name__ == '__main__':

    #if len(sys.argv) < 3:
    img_path   = '.\\pic\\flower.png'
    img_stroke = 'flower-stroke.jpg'
    print "- Unit test for stroke generation. image file=%s" %(img_path)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    stroke = genStroke(img,12, False)
    cv2.imshow('stroke', stroke)
    cv2.waitKey(0)
    #result = np.uint8(stroke * 255)
    #imwrite always write integer value[0,255]
    #cv2.imwrite(img_stroke, result)
    #print "- Write stroke into file. image file=%s" %(img_stroke)
    exit()
    # ignore following code, please.
    img_rgb = cv2.imread(img_path)
    r = img_rgb[:,:,0]
    g = img_rgb[:,:,1]
    b = img_rgb[:,:,2]
    cv2.imshow('r', r)
    cv2.imshow('g', g)
    cv2.imshow('b', b)
    cv2.waitKey(0)
    img_rgb = np.float32(img_rgb) / 255.0
    img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HLS)
    l = img_hls[:,:,1]
    cv2.imshow('l', l)
    cv2.imshow('hls', img_hls)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_HLS2BGR)
    cv2.imshow('bgr', img_bgr)
    cv2.waitKey(0)
    exit() 