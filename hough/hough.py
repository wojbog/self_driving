import numpy as np
import random
from pathlib import Path
import pandas as pd
import cv2

import matplotlib.pyplot as plt



def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def hough_transform(img, crop_top, canny_threshold1, canny_threshold2, apertureSize, blur_kernel_size, rho, theta, threshold):
    img= cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(img, canny_threshold1, canny_threshold2, apertureSize=apertureSize)
    edges[:crop_top, :] = 0  # Crop the top part of the image
    lines = cv2.HoughLines(edges, rho=rho, theta=theta, threshold=threshold)
    return lines[:, 0, :] if lines is not None else None
    

def draw_hough_lines(img, lines):
    if lines is not None:
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img



def hough_transform_p(img, crop_top, canny_threshold1, canny_threshold2, apertureSize, blur_kernel_size, rho, theta, threshold, min_line_length, max_line_gap):
    # crop 50 pixels from the top
    img= cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(img, canny_threshold1, canny_threshold2, apertureSize=apertureSize)
    edges[:crop_top, :] = 0  # Crop the top part of the image
    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    
    return lines[:, 0, :] if lines is not None else None

def draw_hough_lines_p(img, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img
