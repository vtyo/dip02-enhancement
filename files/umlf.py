import numpy as np

#Global variables defining index access on a matrix
ROW = 0
COLUMN = 1

kernel1 = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])

kernel2 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])

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