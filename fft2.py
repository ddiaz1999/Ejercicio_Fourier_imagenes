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
        self.image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # fft visualization
        image_gray_fft_mag = np.absolute(self.image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + np.finfo(np.float32).eps)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        cv2.imshow('fft',image_fft_view)
        cv2.waitKey(10000)

    def LPfilter(self,f):
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        low_pass_mask = np.zeros_like(image_gray)
        high_pass = np.zeros_like(image_gray)
        freq_cut_off = f  # it should less than 1
        half_size = num_rows / 2 - 1  # here we assume num_rows = num_columns
        radius_cut_off = int(freq_cut_off * half_size)
        idx = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        low_pass_mask[idx] = 1

        # filtering via FFT
        fft_filtered = self.image_gray_fft_shift * low_pass_mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)
        cv2.imshow('imagen filtrada pasabajas',image_filtered)
        cv2.waitKey(0)

    def HPfilter(self,f):
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        low_pass_mask = np.zeros_like(image_gray)
        high_pass = np.zeros_like(image_gray)
        freq_cut_off2 = f
        half_size = num_rows / 2 - 1  # here we assume num_rows = num_columns
        radius_cut_off2 = int(freq_cut_off2 * half_size)
        idx = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) > radius_cut_off2
        low_pass_mask[idx] = 1

        # filtering via FFT
        fft_filtered = self.image_gray_fft_shift * low_pass_mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)
        cv2.imshow('imagen filtrada pasa altas',image_filtered)
        cv2.waitKey(0)

    def BPfilter(self,f1,f2):
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        low_pass_mask = np.zeros_like(image_gray)
        high_pass = np.zeros_like(image_gray)
        freq_cut_off1 = f1
        freq_cut_off2 = f2
        half_size = num_rows / 2 - 1  # here we assume num_rows = num_columns
        radius_cut_off1 = int(freq_cut_off1 * half_size)
        radius_cut_off2 = int(freq_cut_off2 * half_size)
        idx1 = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off1
        idx2 = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) > radius_cut_off2
        
        low_pass_mask[idx] = 1

        # filtering via FFT
        fft_filtered = self.image_gray_fft_shift * low_pass_mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)
        cv2.imshow('imagen filtrada pasa banda', image_filtered)
        cv2.waitKey(0)


image_routh = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision\lena.jpg'
image = cv2.imread(image_routh)
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('original gris',image_gray)
#cv2.waitKey(0)

image_class = fft2(image_gray)
image_class.show()
cv2.waitKey(0)

#image_class.LPfilter(0.2)
#image_class.HPfilter(0.6)
#cv2.imshow('image',image_gray)
image_class.BPfilter(0.2,0.6)


#image_gray.show()