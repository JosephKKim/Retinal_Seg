from PIL import Image

# Open the images and convert them to binary format
img1 = Image.open("/home/kkh/research/seg/FR-UNet/save_picture_2class/msp8.png").convert('1')
img2 = Image.open("/home/kkh/research/seg/FR-UNet/save_picture_2class/pre_b8.png").convert('1')

# Get the pixel data for both images
pixels1 = img1.load()
pixels2 = img2.load()

# Initialize the counter
counter = 0

# Loop through all pixels and count the different ones
for x in range(img1.size[0]):
    for y in range(img1.size[1]):
        if pixels1[x,y] != pixels2[x,y]:
            counter += 1

# Print the result
print("Number of different pixels:", counter)
