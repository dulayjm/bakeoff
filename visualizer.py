from PIL import Image
from torchvision import transforms
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np

def visualize(feature_pair, file_pair, out_file, img_shape):
    display_transform = transforms.Compose([
        transforms.Resize(img_shape)
    ])

    fig, axarr = plt.subplots(1,2)

    image = Image.open(file_pair[0]).convert('RGB')
    overlay = getCAM(feature_pair[0])
    axarr[0].imshow(overlay[0], alpha=0.5, cmap='jet')
    axarr[0].imshow(display_transform(image))
    axarr[0].imshow(skimage.transform.resize(overlay[0], img_shape), alpha=0.5, cmap='jet')
    axarr[0].axis('off')

    image = Image.open(file_pair[1]).convert('RGB')
    overlay = getCAM(feature_pair[1])
    axarr[1].imshow(overlay[0], alpha=0.5, cmap='jet')
    axarr[1].imshow(display_transform(image))
    axarr[1].imshow(skimage.transform.resize(overlay[0], img_shape), alpha=0.5, cmap='jet')
    axarr[1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0)
    plt.savefig(out_file, bbox_inches='tight', pad_inches = 0)
    plt.close()

def getCAM(feature_conv):
    nc, h, w = feature_conv.shape
    cam = sum(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]