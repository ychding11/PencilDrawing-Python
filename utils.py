import cv2
import numpy as np

def downImage(img_file, n):
    img = cv2.imread(img_file)
    res = img
    for i in range(n,0,-1):
        res = cv2.pyrDown(res)
        #imwrite always write integer value[0,255]
        result_path = "down%d.jpg"%i
        cv2.imshow(result_path, res)
        cv2.waitKey(0)
    result_path = "%s-down-%d.jpg"%(img_file,n)
    cv2.imwrite(result_path, res)


def genPencilTexture(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)/255.0
    cv2.imshow('original', img)
    cv2.waitKey(1)
    img = np.power(img,1.618)
    cv2.imshow('generated pencil texture', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    downImage('mei5.jpg',2)
    img_path = 'pencil3.jpg'
    genPencilTexture(img_path)