import numpy as np

#Global variables defining index access on a matrix
ROW = 0
COLUMN = 1


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

