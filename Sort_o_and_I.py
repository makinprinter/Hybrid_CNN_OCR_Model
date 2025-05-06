import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt


def calculate_circularity(contour):
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * 3.14159265358979323846 * (area / (perimeter * perimeter))
    return circularity


def Find_circular(contours,circularity_threshold):
    circular_contours = []
    for contour in contours:
        if len(contour) > 1:
            circularity = calculate_circularity(contour)
            if circularity > circularity_threshold:
                circular_contours.append(contour)
    return circular_contours

    
def Is_O_or_I(image_array, max_list=[1,1,1,1], circularity_threshold=0.7):#was [9,3,9,3]
    gray = (image_array * 255).astype(np.uint8) 
    
    thresholded_image = cv2.threshold(gray, 107, 255, cv2.THRESH_BINARY)[1]
    
    
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    O = any(cv2.contourArea(cnt) > circularity_threshold for cnt in contours)
    
    vertical_lines = sum(1 for cnt in contours if cv2.boundingRect(cnt)[2] < max_list[1])
    I = vertical_lines >= 1
    
    horizontal_lines = sum(1 for cnt in contours if cv2.boundingRect(cnt)[3] < max_list[3])
    H_l = horizontal_lines >= 1
    
    return O, I, H_l


def count_valid_lines(grid,max_lenth,max_width):
    grid = grid.astype(int)  # Convert boolean to 1s and 0s
    count = 0

    # Slide over all sets of 3 consecutive rows
    shape = grid.shape
    rows, cols = shape
    for i in range(rows-max_width):  # 26 possible sets of 3 rows in a 28-row grid
        window = grid[i:i+max_width]  # Take 3 consecutive rows
        row_sums = np.sum(window, axis=0)  # Sum along columns
        
        # Only consider columns where all 3 rows are True
        valid_columns = (row_sums == max_width * 255).astype(int)  
        
        # Check for sequences of 15 consecutive Trues in valid_columns
        rolling_sums = np.convolve(valid_columns, np.ones(max_lenth, dtype=int), mode='valid')
        count += np.count_nonzero(rolling_sums == max_lenth)

    return count


def Is_O_or_I_old(image_array,max_list=[1,1,1,1],circularity_threshold=0.7): 
   
    gray = (image_array * 255).astype(np.uint8) 
    contour_image = np.zeros_like(image_array)
    
    
    
    thr =  107
    maxvel = 255
    ret, thresholded_image = cv2.threshold(gray, thr, maxvel, cv2.THRESH_BINARY)
    max_lenth = max_list[0]
    max_width = max_list[1]
    
    dict_lines = {}
    
    l_linesTry2 = count_valid_lines(thresholded_image,max_lenth,max_width)
    
    rotated_image = cv2.rotate(thresholded_image, cv2.ROTATE_90_CLOCKWISE)
    H_max_lenth = max_list[2]
    H_max_width = max_list[3]
    Hl_linesTry2 = count_valid_lines(rotated_image,H_max_lenth,H_max_width)
    
    threshold1 = 50
    threshold2 = 300
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    
    circularity_threshold = 0.7 
    circular_contours = Find_circular(contours,circularity_threshold)
    
    O = False
    H_l = False
    I = False
    if len(circular_contours) >= 1:
        O = True
    if l_linesTry2 >=1 :
        I = True
    if   Hl_linesTry2 >=1 :  
        H_l = True
    return O,I,H_l





