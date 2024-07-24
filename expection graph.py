# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Load the data
# Note: Ensure the file is uploaded and the path is correct
data1 = pd.read_csv("C:/Users/USER/Desktop/설비학회/Dataset_장비이상 조기탐지 AI 데이터셋/data/5공정_180sec/kemp-abh-sensor-2021.09.07.csv")
data2 = pd.read_csv("C:/Users/USER/Desktop/설비학회/Dataset_장비이상 조기탐지 AI 데이터셋/data/5공정_180sec/kemp-abh-sensor-2021.09.06.csv")

# Extract Temp values
real_temp = data1['Temp'].values        
expect_temp = data2['Temp'].values      

nan_inf_indices_real = np.isnan(real_temp) | np.isinf(real_temp)
nan_inf_indices_expect = np.isnan(expect_temp) | np.isinf(expect_temp)

real_temp[nan_inf_indices_real] = np.nanmean(real_temp)
expect_temp[nan_inf_indices_expect] = np.nanmean(expect_temp)

# Ensure both temp arrays are of same length
assert len(real_temp) == len(expect_temp), "데이터의 길이가 서로 다릅니다."

# Create a figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, color='red', zorder=2)               
line2, = ax.plot([], [], lw=2, color=(1, 0.8, 0.8), zorder = 1)  

# Set the axis limits
ax.set_xlim(0, len(real_temp))
ax.set_ylim(min(np.min(real_temp), np.min(expect_temp)) - 1, max(np.max(real_temp), np.max(expect_temp)) + 1)

# Initialize the data arrays
xdata1, ydata1 = [], []
xdata2, ydata2 = [], []

def init():
    line.set_data([], [])
    line2.set_data([], [])
    return line, line2,

def update(frame):
    # Update blue line (data1) only after frame 10 and until the end of real_temp
    if frame >= 10 and frame-10 < len(real_temp):
        xdata1.append(frame-100)                     # 격차조정
        ydata1.append(real_temp[frame])
        line.set_data(xdata1, ydata1)
    
    # Update red line (data2) from frame 0
    xdata2.append(frame)
    ydata2.append(expect_temp[frame])
    line2.set_data(xdata2, ydata2)

    return line, line2,

anim = FuncAnimation(fig, update, frames=range(len(real_temp)+10), init_func=init, blit=True, interval=5) # 그래프 그려지는 속도

plt.xlabel('Time index')
plt.ylabel('Temperature')
plt.title('Temp Variation Over Time')
plt.grid(True)

plt.show()