from xymodel import update_metropolis, vorticity
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
import numpy as np

plt.style.use('ggplot')
plt.rcParams['axes.grid'] = False
plt.rcParams['image.cmap'] = 'hsv'

temp = 1
L = 50
grid = 2 * np.pi * np.random.rand(L, L).astype(np.float32)
x = y = np.arange(L)
fig, ax = plt.subplots()

ax.set_facecolor('w')
img = ax.quiver(x, y, np.cos(grid), np.sin(grid), angles='xy', scale_units='xy', scale=0.4, pivot='mid')
back = ax.matshow(vorticity(grid), cmap='Purples_r')

plt.subplots_adjust(bottom=0.2)
plt.xlim(0, L)
plt.ylim(0, L)
plt.gca().set_aspect('equal', adjustable='box')

def animate_frame(_):
    global grid
    
    grid = update_metropolis(grid, temp)
    img.set_UVC(np.cos(grid), np.sin(grid), grid / (2*np.pi))
    back.set_data(vorticity(grid))
    
    return [img, back]

temp_slider = Slider(
    ax=plt.axes([0.25, 0.1, 0.65, 0.03]),
    label='Temperature', valmin=0.01, valmax=1.5, valinit=temp
)

def slider_upd(_): 
    global temp 
    temp = temp_slider.val

temp_slider.on_changed(slider_upd)

anim = animation.FuncAnimation(fig, animate_frame, frames=300, interval=50, blit=True)

plt.show()
