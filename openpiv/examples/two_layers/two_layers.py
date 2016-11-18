
# coding: utf-8

# In[44]:

# %loadpy tutorial-part1.py


# In[45]:

import sys
sys.path.append('/Users/alex/Documents/OpenPIV/alexlib/openpiv-python')


from openpiv import tools, pyprocess, scaling, validation, filters

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[46]:

frame_a  = tools.imread( 'Run28Oct000000.T000.D000.P000.H002.LA.JPG' )
frame_b  = tools.imread( 'Run28Oct000000.T000.D000.P000.H002.LB.JPG' )


# In[47]:

fig,ax = plt.subplots(1,2)
ax[0].imshow(frame_a,cmap=plt.cm.gray)
ax[1].imshow(frame_b,cmap=plt.cm.gray)


# In[48]:

fig,ax = plt.subplots(1,2)
ax[0].imshow(frame_a[:,:2800],cmap=plt.cm.gray)
ax[1].imshow(frame_b[:,:2800],cmap=plt.cm.gray)


# In[49]:

frame_a = frame_a[:,:2800]
frame_b = frame_b[:,:2800]


# In[50]:

winsize = 64 # pixels
searchsize = 128  # pixels, search in image B
overlap = 32 # pixels
dt = 1 # sec


u0, v0, sig2noise = pyprocess.piv( frame_a, frame_b, window_size=winsize, overlap=overlap, dt=dt, search_size=searchsize, sig2noise_method='peak2peak' )


# In[51]:

x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )


# In[52]:

u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.0 )
print np.nansum((u1 - u0)**2)


# In[53]:

u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=10, kernel_size=2)
print np.nansum((u2 - u1)**2)


# In[54]:

x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 96.52 )
print np.nansum((u3 - u2)**2)


# In[55]:

tools.save(x, y, u3, v3, mask, 'two_layers.txt' )


# In[56]:

tools.display_vector_field('two_layers.txt', scale=10, width=0.0025)


# In[ ]:



