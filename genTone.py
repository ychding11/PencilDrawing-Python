import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import cg

import math
import sys
import os

#for histogram matching
def transferTone(img, verbose = False):
    ho = np.zeros( 256)
    po = np.zeros( 256)
    for i in range(256):
        po[i] = np.sum(img == i)
    po = po / np.sum(po)

    #caculate original cumulative histogram
    ho[0] = po[0]
    for i in range(1,256):
        ho[i] = ho[i - 1] + po[i]

    #use parameter from paper.
    omiga1 = 76
    omiga2 = 22
    omiga3 = 2
    p1 = lambda x : (1 / 9.0) * np.exp(-(255 - x) / 9.0)
    p2 = lambda x : (1.0 / (225 - 105)) * (x >= 105 and x <= 225)
    p3 = lambda x : (1.0 / np.sqrt(2 * math.pi *11) ) * np.exp(-((x - 90) ** 2) / float((2 * (11 **2))))
    p  = lambda x : (omiga1 * p1(x) + omiga2 * p2(x) + omiga3 * p3(x)) * 0.01

    prob = np.zeros(256)
    total = 0
    for i in range(256):
        prob[i] = p(i)
        total = total + prob[i]
    prob = prob / total 

    #caculate new cumulative histogram
    histo = np.zeros(256)
    histo[0] = prob[0]
    for i in range(1, 256):
        histo[i] = histo[i - 1] + prob[i]

    Iadjusted = np.zeros((img.shape[0], img.shape[1]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            histogram_value = ho[img[x,y]]
            i = np.argmin(np.absolute(histo - histogram_value))
            Iadjusted[x, y] = i

    Iadjusted = np.float64(Iadjusted) / 255.0
    if verbose == True:
        cv2.imshow('adjust tone', Iadjusted)
        cv2.waitKey(1)
    J = Iadjusted
    J = cv2.blur(Iadjusted, (3, 3))
    if verbose == True:
        cv2.imshow('blurred adjust tone', J)
        cv2.waitKey(1)
    return J


#compute the pencil tone map
def genTone(img, verbose = False, pencil_texture = 'pencil5.png'):
    height, width = img.shape
    n = height * width
    print "- Image size=%dx%d."%(width, height)
    print "- Extract tone map for orinal image."
    J = transferTone(img, verbose)
    print "- Done."

    P = cv2.imread(pencil_texture, cv2.IMREAD_GRAYSCALE)
    P = np.float64(P) / 255.0
    print "- Pencil texture size=%dx%d."%(P.shape[1], P.shape[0])

    print "- Construct vector."
    P = cv2.resize(P, (height, width))
    P = P.reshape((n, 1))
    logP = np.log(P+2.2204e-15)
    J = J.reshape((n,1))
    logJ = np.log(J+2.2204e-15)
    print "- Done."
    

    print "- Solve formular to find optimal beta."
    theta = 0.2
    b= logJ * logP
    temp = (logP * logP).reshape(1,n)
    A = spdiags(temp,0,n,n)

    #construct A
    e = np.ones(n)
    B = spdiags([e, e], np.array([-width, -1]), n, n)
    up    = np.ones((height, width))
    down  = np.ones((height, width))
    left  = np.ones((height, width))
    right = np.ones((height, width))
    up[0,:]=0
    down[height-1,:]=0
    left[:,0]=0
    right[:,width-1]=0
    C = -(up + down + left + right)
    C = C.reshape((1,n))
    D = spdiags(C, 0, n, n)
    A = A + theta * ((B.transpose())+ B + D)
    
    #solve it
    beta, info = cg(A, b , tol = 1e-6, maxiter = 16)
    print "info=",info
    print "- Done."

    beta = beta.reshape((height, width))
    P    = P.reshape((height, width))
    J    = J.reshape((height, width))
    T    = np.power(P, beta)
    if verbose == True:
        cv2.imshow('resized pencil texture', P)
        cv2.waitKey(1)
        cv2.imshow('beta', beta)
        cv2.waitKey(1)
    cv2.imshow('tone', T)
    cv2.waitKey(1)

    return T


if __name__ == '__main__':
    img_path = '.\\pic\\flower.png'
    print "- Unit test for tone generation. image file=%s" %(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original image', img)
    cv2.waitKey(1)
    T = genTone(img,False)
    cv2.imshow('tone', T)
    cv2.waitKey(0)