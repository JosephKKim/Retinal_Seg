import matplotlib.pyplot as plt
import numpy as np



# overlap two images

# load two images
img1 = plt.imread('image1.jpg')
img2 = plt.imread('image2.jpg')

# create a figure and axis objects
fig, ax = plt.subplots()

# display the first image
ax.imshow(img1)

# set the alpha value of the second image
alpha = 0.5

# overlay the second image
ax.imshow(img2, alpha=alpha)

# show the plot
plt.show()
