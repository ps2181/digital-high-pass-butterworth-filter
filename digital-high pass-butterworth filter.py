#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math


# In[3]:


f_sample = 3500
f_pass = 1050
f_stop = 600
fs = 0.5
wp = f_pass/(f_sample/2)
ws = f_stop/(f_sample/2)
Td = 1
g_pass = 1
g_stop = 50


# In[5]:


omega_p = (2/Td)*np.tan(wp/2)
omega_s = (2/Td)*np.tan(ws/2)
N, Wn = signal.buttord(omega_p, omega_s,g_pass,g_stop,analog=True)
print("Order of the Filter=", N)
print("Cut-off frequency={:3f} rad/s ".format(Wn))

b, a = signal.butter(N, Wn,'high',True)
z, p = signal.bilinear(b, a, fs)

w, h = signal.freqz(z, p, 512)


# In[6]:


plt.semilogx(w, 20*np.log10(abs(h)))
plt.xscale('log')

plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0,0.1)

plt.grid(which='both',axis='both')
plt.axvline(100,color='green')
plt.show()


# In[7]:


imp = signal.unit_impulse(40)
c, d = signal.butter(N, 0.5)
response = signal.lfilter(c, d, imp)

plt.stem(np.arange(0,40),imp,markerfmt='D', use_line_collection=True)
plt.stem(np.arange(0,40),response,use_line_collection=True)
plt.grid(True)
plt.show()


# In[8]:


fig,ax1 = plt.subplots()

ax1.set_title('Digital filter frequency response')
ax1.set_ylabel('Angle(radians)',color='g')
ax1.set_xlabel('Frequency [Hz]')

angles = np.unwrap(np.angle(h))
ax1.plot(w/2*np.pi,angles,'g')
ax1.grid()
ax1.axis('tight')

plt.show()


# In[ ]:




