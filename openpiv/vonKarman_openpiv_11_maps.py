#!/usr/bin/env python
# coding: utf-8

# # compare the OpenPIV Python with PIVLab
# 
# Analysis of the Karman images
# final int area 6 pixels and 50% overlap, 
# vector validation is allowed, but no smoothing after the last correlation. 
# Only the circle in the middle must be masked, not the shadows.

# Then we can compare the vorticity maps (color bar scale of uncalibrated data -0.3 1/frame until +0.3 1/frame, 
# color map preferably "parula", but "jet" is also ok). That might give an idea about the "quality"...?FFT window deformation
# Pass1: 64x64 px with 50% overlap
# Pass2: 32x32 px with 50% overlap
# Pass3: 16x16 px with 50% overlap
# Pass4: 6x6 px with 50% overlap
# Gauss2x3-point subpixel estimator
# Correlation quality: Extreme
# In[ ]:


# get_ipython().run_line_magic('reload_ext', 'watermark')
# get_ipython().run_line_magic('watermark', '-v -m -p numpy,openpiv')


# In[ ]:


from openpiv import windef

import numpy as np
import os
from time import time

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)


# In[ ]:


settings = windef.Settings()

# 'Data related settings'
# Folder with the images to process
settings.filepath_images = './examples/test9/'
# Folder for the outputs
settings.save_path = '.'
# Root name of the output Folder for Result Files
settings.save_folder_suffix = 'Test_1'
# Format and Image Sequence
settings.frame_pattern_a = 'karman_16Hz_000_A.jpg'
settings.frame_pattern_b = 'karman_16Hz_000_B.jpg'

'Region of interest'
# (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
# settings.ROI = 'full'
settings.ROI = (200,400,500,900)

settings.deformation_method = 'second image' #'symmetric' # or 'second image'


settings.num_iterations = 2  # select the number of PIV passes

# add the interrogation window size for each pass. 
# For the moment, it should be a power of 2 
settings.windowsizes=(64, 32, 16, 8, 6)
settings.overlap=(32, 16, 8, 4, 3)

# settings.windowsizes = (128, 64, 32, 16, 8) # if longer than n iteration the rest is ignored
# The overlap of the interroagtion window for each pass.
# settings.overlap = (64, 32, 16, 8, 4) # This is 50% overlap


# Has to be a value with base two. In general window size/2 is a good choice.
# methode used for subpixel interpolation: 'gaussian','centroid','parabolic'
settings.subpixel_method = 'gaussian'

# order of the image interpolation for the window deformation
settings.interpolation_order = 2
settings.scaling_factor = 1  # scaling factor pixel/meter
settings.dt = 1  # time between to frames (in seconds)
'Signal to noise ratio options (only for the last pass)'
# It is possible to decide if the S/N should be computed (for the last pass) or not
settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
settings.sig2noise_threshold = 1.1
# method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
settings.sig2noise_method = 'peak2peak'
# select the width of the masked to masked out pixels next to the main peak
settings.sig2noise_mask = 2
settings.do_sig2noise_validation = False
# If extract_sig2noise==False the values in the signal to noise ratio
# output column are set to NaN

# only effecting the first pass of the interrogation the following passes
# in the multipass will be validated

'Output options'
# Select if you want to save the plotted vectorfield: True or False
settings.save_plot = False
# Choose wether you want to see the vectorfield or not :True or False
settings.show_plot = False
settings.scale_plot = 200  # select a value to scale the quiver plot of the vectorfield
# run the script with the given settings


#  'Processing Parameters'
settings.correlation_method = 'circular'  # 'circular' or 'linear'
settings.normalized_correlation = False

# 'vector validation options'
# choose if you want to do validation of the first pass: True or False
settings.validation_first_pass = True


settings.filter_method = 'localmean'
# maximum iterations performed to replace the outliers
settings.max_filter_iteration = 4
settings.filter_kernel_size = 2  # kernel size for the localmean method

settings.replace_vectors = False

settings.MinMax_U_disp = (-10, 10)
settings.MinMax_V_disp = (-10, 10)

# The second filter is based on the global STD threshold
settings.std_threshold = 3  # threshold of the std validation

# The third filter is the median test (not normalized at the moment)
settings.median_threshold = 3  # threshold of the median validation
# On the last iteration, an additional validation can be done based on the S/N.
settings.median_size = 2 #defines the size of the local median, it'll be 3 x 3


settings.dynamic_masking_method = 'intensity'
settings.dynamic_masking_threshold = 0.1
settings.dynamic_masking_filter_size = 21

# New settings for version 0.23.2c
settings.image_mask = True

# Smoothing after the first pass
settings.smoothn = False #Enables smoothing of the displacemenet field
settings.smoothn_p = 0.5 # This is a smoothing parameter


# In[ ]:


import glob
file_list = sorted(glob.glob('examples/test9/karman_16Hz_*.jpg'))
file_list = file_list[-2:]
print(file_list)


# In[ ]:


# N = 1
# counter = 0
# for a, b in zip(file_list[0:2*N+1:2],file_list[1:2*N+2:2]):
#     print(a,b)
#     settings.frame_pattern_a = a
#     settings.frame_pattern_b = b
#     settings.save_folder_suffix = f'{counter}'
windef.piv(settings)
#     counter += 1


# In[ ]:


# from pivpy import pivpy, io, graphics
# import xarray as xr


# # In[ ]:


# import glob
# file_list = sorted(glob.glob('./Open_PIV_results_6_0/*.txt'))

# data = []
# frame = 0
# for f in file_list:
#     data.append(io.load_txt(f,frame=frame))
#     frame += 1
    
# data = xr.concat(data, dim='t')
# data.attrs['units']= ['pix','pix','pix/dt','pix/dt']


# # In[ ]:


# data.piv.vorticity()


# # In[ ]:


# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(12,5))
# # for ax in axs:
# s = ax.pcolor(data.x,data.y,data.w.T.isel(t=0), shading='nearest', vmin=-.3, vmax=.3)
# ax.invert_yaxis()
# ax.set_aspect(1)
# fig.colorbar(s, ax=ax,)


# # In[ ]:


# data.w.T.values


# In[ ]:




