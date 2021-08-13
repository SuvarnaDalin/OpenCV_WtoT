# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:26:02 2021

@author: suvarna
"""

########################## LIBRARIES & MODULES #############################

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import base64

########################## DEFINE FUNCTIONS ################################
# Function to covert Red background image to Transparent background image
def RedToTransparent(red):
    
    img = red.convert("RGBA")    
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return(img)

# Function to download Image
def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

############################################################################
########################## CODE BEGINS HERE ################################
############################################################################

# Give a title
st.title('TRANSPARENT BACKGROUND IMAGE GENERATOR')

# Upload the images
st.markdown('**White Background Image**')
img_data = st.file_uploader(label='Load White Background Image', type=['png', 'jpg', 'jpeg'])

if img_data is not None:
    
    # Display uploaded image
    uploaded_img = Image.open(img_data)
    st.title('Image with White Background')
    st.image(uploaded_img)

    img = np.array(uploaded_img)
    #img = Image.fromarray(uploaded_img)
    #img = cv2.imread(img_data)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(image_gray, 254, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Contour Mapping
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #Masking
    mask = np.zeros(thresh.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    # Draw contours
    masked = cv2.drawContours(mask, [largest_areas[-1]],0,(255, 255, 255, 255),-1)
    

    
    # GRABCUT the image from the mask
    
    img11 = np.asarray(uploaded_img)
    img1 = cv2.cvtColor(img11, cv2.COLOR_RGBA2RGB)
    newmask = masked
    
    # wherever it is marked white (sure foreground), change mask=1
    # wherever it is marked black (sure background), change mask=0
    mask_ch = np.zeros(img.shape[:2],np.uint8)
    mask_ch[newmask == 0] = 0
    mask_ch[newmask == 255] = 1
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    cv2.grabCut(img1,mask_ch,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    #cv2.grabCut(img1,mask_ch,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
    mask1 = np.where((mask_ch==2)|(mask_ch==0),0,1).astype('uint8')
    img1 = img1*mask1[:,:,np.newaxis]
    
    # Converting BGR To RGB and saving
    #final = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    final = img1
    black_image = Image.fromarray(final.astype(np.uint8))
    
    # Creating a Red background
    bckgrnd = np.full_like(final, (255, 0, 0), dtype=np.uint8)
    
    mask = masked
    
    # apply inverse mask to colored background image
    bckgrnd_masked = cv2.bitwise_or(bckgrnd, bckgrnd, mask=255-mask)
    
    # combine the two
    result = cv2.add(final, bckgrnd_masked)
    result_image = Image.fromarray(result.astype(np.uint8))
    
    #Generate output image
    # Converting Red to transparent background and saving result
    out_image = RedToTransparent(result_image)
    
    #display output image
    st.title('Image with Transparent Background')
    st.image(out_image) 
    
    ## Original image came from cv2 format, fromarray convert into PIL format
    img_file = 'Transparent.png'
    st.markdown(get_image_download_link(out_image,img_file,'Download '+img_file), unsafe_allow_html=True)
