from PIL import Image
from torchvision import transforms
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np

def visualize(features1, file1, features2, file2, size):
    display_transform = transforms.Compose([
        transforms.Resize((256,256))
    ])

    image = Image.open(file1).convert('RGB')
    overlay = getCAM(features1)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax1.imshow(overlay[0], alpha=0.5, cmap='jet')
    ax1.imshow(display_transform(image))
    ax1.imshow(skimage.transform.resize(overlay[0], size), alpha=0.5, cmap='jet')
    ax1.axis('off')

    image = Image.open(file2).convert('RGB')
    overlay = getCAM(features2)
    ax2.imshow(overlay[0], alpha=0.5, cmap='jet')
    ax2.imshow(display_transform(image))
    ax2.imshow(skimage.transform.resize(overlay[0], size), alpha=0.5, cmap='jet')
    ax2.axis('off')
    plt.show()
    plt.close()

def getCAM(feature_conv):
    nc, h, w = feature_conv.shape
    cam = sum(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]