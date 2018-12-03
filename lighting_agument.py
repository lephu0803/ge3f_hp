import cv2
import numpy as np 
import PIL 

class DataHelper(object):
    def __init__(self, file_path)
        self.file_path = file_path
        self.w, self.h = __read_img(file_path[0])
    def __read_image(images):
        
    def next_batch(self, file_path):
        

def __read_image(images, batch_size):
    for img in images

def add_gaussian_noise(X_img):
    gaussian_noise_imgs = []
    row, col, _ = X_img.shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
    gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
    gaussian_img = np.array(gaussian_img, dtype=np.float32)
    # gaussian_noise_imgs.append(gaussian_img)
    # gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_img

add_gaussian_noise('')