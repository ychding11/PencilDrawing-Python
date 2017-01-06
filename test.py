from genTone   import genTone
from genStroke import genStroke
import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img_path = '.\\pic\\oldman.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    stroke = genStroke(img,8, False)
    tone   = genTone(img,False)
    imgPencil = stroke * tone

    # save the image data into disk cause precision loss
    # and affects image quality.
    #imgTemp = np.uint8(imgPencil * 255)
    #imwrite always write integer value[0,255]
    #cv2.imwrite('pencil-mei.jpg', imgTemp)   

    #try to convert into colored pencil drawing
    img_rgb_original = cv2.imread(img_path)
    img_yuv = cv2.cvtColor(img_rgb_original, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = imgPencil * 255
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB) 
    
    #show image
    #img_yuv   = cv2.equalizeHist(img_yuv * 255) #enhance contrast sometime
    #imgPencil = cv2.equalizeHist(imgPencil * 255) #enhance contrast sometime
    cv2.imshow('rgb',img_rgb_original) 
    cv2.imshow('tone',tone) 
    cv2.imshow('pencil drawing',imgPencil) 
    cv2.imshow('color pencil draw',img_rgb) 
    cv2.waitKey(0)