import numpy as np
import bf

#Global variables defining index access on a matrix
ROW = 0
COLUMN = 1

#Function that normalizes an image into the scale 0-255
def normalize(img):
    return ((img-np.min(img))*255.0)/(np.max(img)-np.min(img))

#Function that creates a 1D filter matrix
def create_filter_vector(size):
    vector = []
    center = int((size/2)-1)
    if (size%2):
        center = int((size/2))

    value = float((-1)*center)

    for x in range(size):
        element = []
        element.append(value)
        vector.append(np.array(element))
        value = value + 1.0

    return np.array(vector)

#Function that applies the Vignette filter into an image based on the sigma values
def vignette_filter(img, sigma_row, sigma_col):
    #w_col is being transposed because the bf.gaussian_kernel function is returning a column matrix already
    #That's why t_w_row doesn't need to be transposed and w_col needs to
    t_w_row = bf.gaussian_kernel(create_filter_vector(img.shape[ROW]), sigma_row)
    w_col = np.transpose(bf.gaussian_kernel(create_filter_vector(img.shape[COLUMN]), sigma_col))

    return normalize(img*(np.matmul(t_w_row, w_col)))

