from PIL import Image
from torchvision import transforms
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

def map_features(outputs, labels, out_file):
    # create array of column for each feature output
    feat_cols = ['feature'+str(i) for i in range(outputs.shape[1])]
    
    # make dataframe of outputs -> labels
    df = pd.DataFrame(outputs, columns=feat_cols)
    df['y'] = labels
    df['labels'] = df['y'].apply(lambda i: str(i))
    
    # clear outputs and labels
    outputs, labels = None, None
    
    # creates an array of random indices from size of outputs
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    
    num_examples = 10000
    
    df_subset = df.loc[rndperm[:num_examples],:].copy()
    data_subset = df_subset[feat_cols].values
    
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(data_subset)
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    plt.figure(figsize=(16,10))
    plt.scatter(
        x=df_subset["tsne-2d-one"],
        y=df_subset["tsne-2d-two"],
        c=df_subset["y"]
    )
    plt.savefig(out_file, bbox_inches='tight', pad_inches = 0)
    plt.close()

def visualize(feature_pair, file_pair, out_file, img_shape):
    display_transform = transforms.Compose([
        transforms.Resize(img_shape)
    ])

    fig, axarr = plt.subplots(2,2)

    image = Image.open(file_pair[0]).convert('RGB')
    overlay = getCAM(feature_pair[0])
    axarr[0,0].imshow(overlay, alpha=0.5, cmap='jet')
    axarr[0,0].imshow(display_transform(image))
    axarr[0,0].imshow(skimage.transform.resize(overlay, img_shape), alpha=0.5, cmap='jet')
    axarr[0,0].axis('off')
    # display normal image
    axarr[1,0].imshow(display_transform(image))
    axarr[1,0].axis('off')

    image = Image.open(file_pair[1]).convert('RGB')
    overlay = getCAM(feature_pair[1])
    axarr[0,1].imshow(overlay, alpha=0.5, cmap='jet')
    axarr[0,1].imshow(display_transform(image))
    axarr[0,1].imshow(skimage.transform.resize(overlay, img_shape), alpha=0.5, cmap='jet')
    axarr[0,1].axis('off')
    # display normal image
    axarr[1,1].imshow(display_transform(image))
    axarr[1,1].axis('off')

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
    return cam_img