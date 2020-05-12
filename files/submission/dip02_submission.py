#Author: Victor Antonio de Oliveira
#nUSP: 9791326

#Git repository: https://github.com/vtyo/dip02-enhancement

#Assignment 2 : Image Enhancement and Filtering
import numpy as np
import imageio

#Global variables defining index access on a matrix
ROW = 0
COLUMN = 1

kernel1 = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])

kernel2 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])

#Function that creates the filter matrix based on the Euclidian Distance to the center element
def create_filter_matrix(filter_size):
    matrix = []
    x_elem = int(filter_size/2*(-1))
    y_elem = int(filter_size/2*(-1))
    
    for x in range(filter_size):
        row = []
        for y in range(filter_size):
            row.append(np.sqrt(np.power(x_elem, 2) + np.power(y_elem, 2)))
            y_elem = y + 1
        matrix.append(row)
        x_elem = x + 1
    return np.array(matrix)    

#Function that calculates the Gaussian Kernel given an element and a sigma value
def gaussian_kernel(x, sigma):
    return (
        1/(2*(np.pi*np.power(sigma, 2)))*
        np.exp(((-1)/2)*(np.power(x, 2)/np.power(sigma, 2)))
    )

#Function that processes a Bilateral filter into an image applying a spatial Gaussian filter and a range Gaussian
def bilateral_filter(img, spatial, sigma_r):
    result_img = []
    for img_x, img_row  in enumerate(img):
        result_row = []
        for img_y, img_pixel in enumerate(img_row):
            
            pixel = 0
            wp = 0
            
            #Center element's index
            s_center = int(spatial.shape[ROW]/2)
            
            #Neighbor's row index
            neighbor_x = img_x - s_center
            for s_row in spatial:
                #Neighbor's column index
                neighbor_y = img_y - s_center
                for gsi in s_row: 
                    #Neighbor = 0 if outside image matrix   
                    neighbor = 0
                    #Otherwise, it'll be its actual value
                    if(
                        neighbor_x >= 0 and 
                        neighbor_y >= 0 and 
                        neighbor_x < img.shape[ROW] and
                        neighbor_y < img.shape[COLUMN]
                    ):
                        neighbor = img[neighbor_x][neighbor_y]
                    
                    gri = (
                            gaussian_kernel(
                            neighbor - img_pixel, sigma_r
                            ) 
                    )

                    wi = gri*gsi

                    wp = wp + wi 
                    pixel = pixel + (wi*neighbor)
                    
                    neighbor_y = neighbor_y + 1
                neighbor_x = neighbor_x + 1
            
            result_row.append(pixel/wp)
        result_img.append(result_row)
    return np.array(result_img)

#Function that normalizes an image into the scale 0-255
def normalize(img):
    return ((img-np.min(img))*255)/(np.max(img)-np.min(img))

#Function that applies an Unsharp mask into an image using a Laplacian filter 
def unsharp_mask(img, param_c, k_number):
    kernel = []
    if (k_number == 1):
        kernel = kernel1
    else:
        kernel = kernel2
    
    result_img = []
    for img_x, img_row  in enumerate(img):
        result_row = []
        for img_y, img_pixel in enumerate(img_row):
    
            pixel = 0
            k_center = 1

            #Neighbor's row index
            neighbor_x = img_x - k_center
            for k_row in kernel:
                #Neighbor's column index
                neighbor_y = img_y - k_center
                for k_elem in k_row:    
                    #Neighbor = 0 if outside image matrix   
                    neighbor = 0
                    #Otherwise, it'll be its actual value
                    if(
                        neighbor_x >= 0 and 
                        neighbor_y >= 0 and 
                        neighbor_x < img.shape[ROW] and
                        neighbor_y < img.shape[COLUMN]
                    ):
                        neighbor = img[neighbor_x][neighbor_y]
                    
                    pixel = pixel + (neighbor*k_elem)
                    
                    neighbor_y = neighbor_y + 1
                neighbor_x = neighbor_x + 1
            
            
            result_row.append(pixel)
        result_img.append(result_row)

    result_img = normalize(np.array(result_img))
    result_img = (param_c*result_img) + img
    
    return normalize(result_img)

#Function that normalizes an image into the scale 0-255
def normalize(img):
    return ((img-np.min(img))*255)/(np.max(img)-np.min(img))

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
    #w_col is being transposed because the gaussian_kernel function is returning a column matrix already
    #That's why t_w_row doesn't need to be transposed and w_col needs to
    t_w_row = gaussian_kernel(create_filter_vector(img.shape[ROW]), sigma_row)
    w_col = np.transpose(gaussian_kernel(create_filter_vector(img.shape[COLUMN]), sigma_col))

    return normalize(img*(np.matmul(t_w_row, w_col)))

filename = str(input()).rstrip()
method = int(input())
save = int(input())

input_img = imageio.imread(filename)

#We'll use auxiliar numpy arrays to store the images(input and output) values into int32 format
#Using numpy, it is possible to procces the output values without iterating the arrays with a for loop
input_img_array = np.array(input_img).astype(np.int32)
output_img_array = np.array([])

#Bilateral Filtering
if method == 1:
    filter_size = int(input())
    sigma_s = float(input())
    sigma_r = float(input())
    
    filter_matrix = create_filter_matrix(filter_size)
    spatial = gaussian_kernel(filter_matrix, sigma_s)
    output_img_array = bilateral_filter(input_img_array, spatial, sigma_r)
    
#Unsharp Masking
if method == 2:
    param_c = float(input())
    k_number = int(input())

    output_img_array = unsharp_mask(input_img_array, param_c, k_number)

#Vignette Filtering
if method == 3:
    sigma_row = float(input())
    sigma_col = float(input())

    output_img_array = vignette_filter(input_img_array, sigma_row, sigma_col)
    
rse = np.sqrt(np.sum(np.power((output_img_array - input_img_array), 2)))

print("%.4f" % rse)

output_img = output_img_array.astype(np.uint8) 
    
if save == 1:
    imageio.imwrite('output_img.png', output_img)

