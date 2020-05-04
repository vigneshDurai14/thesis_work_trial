import cv2
import numpy as np
import matplotlib.pyplot as plt

'exec(%matplotlib inline)'

def display_text_img(img,cmap='gray'):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()
    
def get_contour_center(contour):
    M = cv2.moments(contour)
    cx=-1
    cy=-1
    if (M['m00']!=0):
        cx= int(M['m10']/M['m00'])
        cy= int(M['m01']/M['m00'])
    return cx, cy

def conveyor_isolation(image):
    image_copy = image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # APPLYING THRESHOLD.
    ret,binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15,-2) 

    # detection of horizontal and vertical lines
    V = cv2.Sobel(binary, cv2.CV_8U, dx=1, dy=0)
    H = cv2.Sobel(binary, cv2.CV_8U, dx=0, dy=1)

    # grabbing the whole horizontal and vertical bars
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14,14))
    V = cv2.morphologyEx(V, cv2.MORPH_DILATE, kernel, iterations = 3)
    H = cv2.morphologyEx(H, cv2.MORPH_DILATE, kernel, iterations = 3)
    rows,cols = image.shape[:2]
    mask_hor = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_ver = np.zeros(image.shape[:2], dtype=np.uint8)

    # drawing vertical line 
    contours = cv2.findContours(V, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.drawContours(mask_ver, [cnt], -1, 255,-1)

    # drawing horizontal line        
    contours = cv2.findContours(H, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.drawContours(mask_hor, [cnt], -1, 255,-1)

    # isolation technique --> erosion, dilation, morphological gradient, blurring apply anyone of this technique in that real images only in particular rows and columns which we grab the conveyor part alone.

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    erosion_ver = cv2.morphologyEx(image_copy, cv2.MORPH_DILATE, kernel, iterations = 3)
    erosion_hor = cv2.morphologyEx(image_copy, cv2.MORPH_DILATE, kernel, iterations = 3)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12,12))
    #erosion_ver = cv2.morphologyEx(tire_img_1_BGR, cv2.MORPH_GRADIENT, kernel, iterations = 3)
    #erosion_hor = cv2.morphologyEx(tire_img_1_BGR, cv2.MORPH_GRADIENT, kernel, iterations = 3)

    #kernel = np.ones((15,15),np.uint8)
    #erosion_ver = cv2.erode(tire_img_1_BGR,kernel,iterations = 3)
    #erosion_hor = cv2.erode(tire_img_1_BGR,kernel,iterations = 3)

    #kernel = np.ones((15,15),np.uint8)
    #erosion_ver = cv2.dilate(tire_img_1_BGR,kernel,iterations = 3)
    #erosion_hor = cv2.dilate(tire_img_1_BGR,kernel,iterations = 3)

    #erosion_ver = cv2.medianBlur(tire_img_1_BGR,25)
    #erosion_hor = cv2.medianBlur(tire_img_1_BGR,25)

    #erosion_ver = cv2.GaussianBlur(tire_img_1_BGR,(17,17),20)
    #erosion_hor = cv2.GaussianBlur(tire_img_1_BGR,(17,17),20)

    (rows_ver, cols_ver) = np.where(mask_ver != 0)
    image_copy[mask_ver == 255] = erosion_ver[rows_ver, cols_ver]
    (rows_hor, cols_hor) = np.where(mask_hor != 0)
    image_copy[mask_hor == 255] = erosion_hor[rows_hor, cols_hor]        
    return image_copy

def watershed_histogram(original_image, image = None):
    
    if image == None: #IF IMAGE IS NOT LOADED WITH CONVEYOR part.
        image_copy = original_image.copy()
    else: # if image is loaded with conveyor part.
        image_copy = image.copy()
    
    HSV_img_1 = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    
    # grabbing the value channel and equalize the histogram. and replace it back
    HSV_img_1[:,:,2]  = cv2.equalizeHist(HSV_img_1[:,:,2])   
    
    # converting the back to RGB
    HSV_img_1_eqhist = cv2.cvtColor(HSV_img_1, cv2.COLOR_HSV2BGR) 
    
    img_blur_watershed = cv2.medianBlur(HSV_img_1_eqhist,5)
    img_grayscale_watershed = cv2.cvtColor(img_blur_watershed,cv2.COLOR_BGR2GRAY)
    
    rep_img_watershed, img_binary_threshold_otsu_watershed = cv2.threshold(img_grayscale_watershed,90,255, cv2.THRESH_BINARY_INV+cv2.THRESH_TOZERO) #you can vary the threshold parameter, now it's mentioned as 90 - 255
     
    #adaptiveThreshold_gas = cv2.adaptiveThreshold(img_grayscale_watershed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,55,13)
    
    #display_text_img(img_binary_threshold_otsu_watershed)
    kernel_watershed = np.ones((7,7),np.uint8)
    opening_watershed = cv2.morphologyEx(img_binary_threshold_otsu_watershed,cv2.MORPH_CLOSE,kernel_watershed,3)
    #opening_watershed = cv2.morphologyEx(img_binary_threshold_otsu_watershed,cv2.MORPH_GRADIENT,kernel_watershed,3)
    
    # applying distance transfrom --> we can differentiate the borders of image between the foreground and background.
    distance_transform = cv2.distanceTransform(opening_watershed, cv2.DIST_L2,5)
    #display_text_img(distance_transform)
    
    ret_fore, sure_fg = cv2.threshold(distance_transform,0.0012*distance_transform.max(), 255,0)
    #display_text_img(sure_fg)
    
    sure_bg = cv2.dilate(opening_watershed,kernel_watershed,3)
    #display_text_img(sure_bg)
    
    # so convert the sure_fg to uint8
    sure_fg = np.uint8(sure_fg)
    # to find unknow region 
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # to indentify labelling marker 6 fg and seed for the watershed alogrithm find the segments
    ret_mark, markers = cv2.connectedComponents(sure_fg)
    
    HSV_img_1_eqhist_copy = image_copy.copy()
    tire_image = original_image.copy()
    external_coutours = np.zeros(HSV_img_1_eqhist_copy.shape)
    markers = markers + 1
    markers[unknown==255] = 0
    
    # applying the watershed to isolate objects in the image from the background and for finding sure foreground area
    markers_watershed = cv2.watershed(HSV_img_1_eqhist_copy,markers)
    #display_text_img(markers_watershed)
    
    # applying the coutour for finding the circle.
    image_sep_wat, sep_find_coutour_wat, hierarchy_sep_wat = cv2.findContours(markers_watershed.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(sep_find_coutour_wat)):
        if hierarchy_sep_wat[0][i][3] == -1:
            #cv2.drawContours(img_1_BGR_copy,sep_find_coutour_wat,i,255,10) 
            area = cv2.contourArea(sep_find_coutour_wat[i])
            perimeter= cv2.arcLength(sep_find_coutour_wat[i], True)
            ((x, y), radius) = cv2.minEnclosingCircle(sep_find_coutour_wat[i])
            if radius > 500 and radius <= 1000: #you need to vary for each image because each image taken from different position
                cv2.drawContours(HSV_img_1_eqhist_copy,sep_find_coutour_wat,i,255,10) 
                cx, cy = get_contour_center(sep_find_coutour_wat[i])
                cv2.circle(external_coutours, (cx,cy),(int)(radius),(0,255,255),10)
                cv2.circle(tire_image, (cx,cy),(int)(radius),(0,255,255),10)
                #print ("radius: {}, Area: {}, Perimeter: {}".format(radius, area, perimeter))
    return HSV_img_1_eqhist_copy

def main():
    tire_img_1_BGR = cv2.imread('tire_images/img_1.jpg')
    
    #tire_img_1_BGR = cv2.imread('tire_images/img_4_con.jpg')
    # don't use the conveyor images --> conveyor_isolation function is not perfect but techniques used in this seems to be good but need to set parameters and images also not good    
    #conveyor_isolation_image =  conveyor_isolation(tire_img_1_BGR)
    #display_text_img(conveyor_isolation_image)
    #tire_detection = watershed_histogram(tire_img_1_BGR, conveyor_isolation_image)
    
    tire_detection = watershed_histogram(tire_img_1_BGR)
    display_text_img(tire_detection)

if __name__ == '__main__':
    main()
    
