import cv2
import numpy as np
import os

class fft2:
    def __init__(self,image_gray):
        self.image_gray = image_gray

    def show(self):
        self.image1 = image_gray.copy()

        if self.image_gray.shape[0] != self.image_gray.shape[1]:
            size = max(self.image_gray.shape[0],self.image_gray.shape[1])
            self.image1 = np.zeros((size,size),np.uint8)
            self.image1[:self.image_gray.shape[0],:self.image_gray.shape[1]] = self.image_gray

        image_gray_fft = np.fft.fft2(self.image1)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # fft visualization
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + np.finfo(np.float32).eps)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        cv2.imshow('fft',image_fft_view)
        cv2.waitKey(10000)

    #def LPfilter(self,f):

image_routh = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision\lena.jpg'
image = cv2.imread(image_routh)
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

image_class = fft2(image_gray)
image_class.show()
#cv2.imshow('image',image_gray)
#cv2.waitKey(0)


#image_gray.show()