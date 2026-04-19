import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
import matplotlib.pyplot as plt


conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network
weights = conv_net.conv_stack[0].weight.detach().numpy()


# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, 
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be 
# between 0 and 1 before plotting.

fig, axes = plt.subplots(4, 8, figsize=(12,6))
for i, ax in enumerate(axes.flat):
    kernel = weights[i, 0, :, :]
    # normalize to [0, 1]
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    ax.imshow(kernel, cmap='gray')
    ax.axis('off')
plt.suptitle('Kernels From First Conv Layer')
plt.tight_layout()
plt.savefig('kernel_grid.png') # save grid to file and add image to pdf report
plt.show()

# axes.flat lets you iterate over 2D grid of subplots as if it were a flat list


# Apply the kernel to the provided sample image.

img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0					# Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image
# this applies the first conv layer which I think means the same thing
output = conv_net.conv_stack[0](img)


# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.

output = output.squeeze(0)
output = output.unsqueeze(1)


# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.

# convert from tensor to numpy
output = output.detach().numpy()

# using same names which will overwrite but that's ok
fig, axes = plt.subplots(4, 8, figsize=(12,6))
for i, ax in enumerate(axes.flat):
    kernel = output[i, 0, :, :]
    # normalize to [0, 1]
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    ax.imshow(kernel, cmap='gray')
    ax.axis('off')
plt.suptitle('Sample Image After Kernel Application')
plt.tight_layout()
plt.savefig('image_transform_grid.png') # save grid to file and add image to pdf report
plt.show()



# Create a feature map progression.
# You can manually specify the forward pass order or programatically track each activation through the forward pass of the CNN.

# pass image through all layers
post_conv1 = conv_net.conv_stack[0](img)
post_relu1 = conv_net.conv_stack[1](post_conv1)
post_pool = conv_net.conv_stack[2](post_relu1)
post_conv2 = conv_net.conv_stack[3](post_pool)
post_relu2 = conv_net.conv_stack[4](post_conv2)

img_stages = [img, post_conv1, post_relu1, post_pool, post_conv2, post_relu2]
labels = ['Original', 'Conv 1', 'ReLU 1', 'MaxPool', 'Conv 2', 'ReLU 2']

# using same names which will overwrite but that's ok
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for i, (ax, stage, label) in enumerate(zip(axes.flat, img_stages, labels)):
    feature_map = stage.detach().numpy()[0, 0, :, :]
    # normalize to [0, 1]
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    ax.imshow(feature_map, cmap='gray')
    ax.set_title(label)
    ax.axis('off')
plt.suptitle('Feature Progression')
plt.tight_layout()
plt.savefig('feature_progression.png') # save image to file and add to pdf report
plt.show()