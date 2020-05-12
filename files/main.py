#Author: Victor Antonio de Oliveira
#nUSP: 9791326

#Assignment 2 : Image Enhancement and Filtering
import numpy as np
import imageio
import bf
import umlf
import vf

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
    
    filter_matrix = bf.create_filter_matrix(filter_size)
    spatial = bf.gaussian_kernel(filter_matrix, sigma_s)
    output_img_array = bf.bilateral_filter(input_img_array, spatial, sigma_r)
    
#Unsharp Masking
if method == 2:
    param_c = float(input())
    k_number = int(input())

    output_img_array = umlf.unsharp_mask(input_img_array, param_c, k_number)

#Vignette Filtering
if method == 3:
    sigma_row = float(input())
    sigma_col = float(input())

    output_img_array = vf.vignette_filter(input_img_array, sigma_row, sigma_col)
    
rse = np.sqrt(np.sum(np.power((output_img_array - input_img_array), 2)))

print("%.4f" % rse)

output_img = output_img_array.astype(np.uint8) 
    
if save == 1:
    imageio.imwrite('output_img.png', output_img)

