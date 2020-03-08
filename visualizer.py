from PIL import Image
from torchvision import transforms
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np

def visualize(feature_pair, file_pair, out_file, img_shape):
    display_transform = transforms.Compose([
        transforms.Resize(img_shape)
    ])

    fig = plt.figure()

    image = Image.open(file_pair[0]).convert('RGB')
    overlay = getCAM(feature_pair[0])
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(overlay[0], alpha=0.5, cmap='jet')
    ax1.imshow(display_transform(image))
    ax1.imshow(skimage.transform.resize(overlay[0], img_shape), alpha=0.5, cmap='jet')
    ax1.axis('off')

    image = Image.open(file_pair[1]).convert('RGB')
    overlay = getCAM(feature_pair[1])
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(overlay[0], alpha=0.5, cmap='jet')
    ax2.imshow(display_transform(image))
    ax2.imshow(skimage.transform.resize(overlay[0], img_shape), alpha=0.5, cmap='jet')
    ax2.axis('off')

    plt.savefig(out_file, bbox_inches='tight', pad_inches = 0)
    plt.close()

def getCAM(feature_conv):
    nc, h, w = feature_conv.shape
    cam = sum(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]