# %%
from IPython import get_ipython
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array, Normalize
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2hsv
import numpy as np
from pathlib import Path
import os
from PIL import ImageColor
# %% HELPER FUNCTIONS
# -----------------------------------------------------
# File lister:
def file_lister(folder, suffix='jpg'):
    '''List files in folder derived from current work dir.

    listing files with specified suffix in subfolders 
    derived from current work dir or inside current work dir.

    Args:
    ---
        folder: string, name of subfolder to find in if you
                need listing in current dir, just type '.' as
                folder.
        suffix: string, suffix of files, like 'jpg', 'png' etc.
    '''
    basepath = Path(os.getcwd())
    pics_list= list(basepath.glob(f'{folder}/*.{suffix}'))
    return pics_list
        
# Image loader as array
def load_im(path, plugin=None):
    '''Loads image as array.

    takes a string or path-like object and loads image using
    skimage.io.imread into array (like numpy array).

    Args:
    ---
        path: string or path-like object,. path to image.
        plugin: can be GDAL, PIL ... GDAL -> good for GeoTIFF
                default is None.
    '''
    im = imread(path, plugin)
    return im

# Image resize by factor:
def im_resize(im, fact):
    '''Resizes image by factor.

    takes numpy array IMAGE and resizes it by given factor.

    Args:
    ---
        im: image as numpy array.
        fact: integer number as a factor for floor division of image.
    '''
    im = resize(im,
    (im.shape[0] // fact, im.shape[1] // fact),
    anti_aliasing=True)
    return im

def hex2rgb_arr(hex_list):
    rgb_list = [(ImageColor.getcolor(x, 'RGB')) for x in hex_list]
    rgb_arr = np.array(rgb_list).astype(np.uint8)
    return rgb_arr

def show_pal(pal):
    fig, ax = plt.subplots()
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    idxs = np.arange(len(pal)).reshape(3,int(len(pal)/3))
    imshow(pal[idxs])
    counter = 0
    for i in range(3):
        for j in range(int(len(pal)/3)):
            ax.annotate(pal.tolist()[counter],(j,i), ha='center')
            counter += 1

def save_im(name, arr, pal):
    imsave(name, pal[arr])

def label_predict(im_arr, cls_type, nc, n_dim, cov_t='full'):
    '''Performs selected clustering method and returns img array

    takes image array(like-np.array) reshapes it to corretct shape
    for scikit-clustering, makes computation and returns labels as
    np.array with shape of input image.

    Args:
    ---
        im_arr:  input image as numpy array
        cls_type: 1 is GaussianMixture, 2 is Birch -- clusternig 
        nc: number of components/clusters
        n_dim: dimension of input raster, usually int "3" for RGB image
        cov_t: only for GaussianMixture,
                covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
    '''
    # preprocess image to coorect shape for clustering
    x = im_arr.shape[0]
    y = im_arr.shape[1]
    im = im_arr.reshape(-1, n_dim)
    # GaussianMixture
    if cls_type == 1:
        cls = GaussianMixture(n_components=nc, covariance_type=cov_t, random_state=0).fit(im)
        lbl = cls.predict(im)
        lbl = lbl.reshape(x, y)
        return lbl
    # Birch
    elif cls_type == 2:
        cls = Birch(n_clusters=nc).fit(im)
        lbl = cls.predict(im)
        lbl = lbl.reshape(x, y)
        return lbl
    else:
        print('Something wrong happened with "label_predict')
# -----------------------------------------------------
co_pal = np.array([
    [188,  14,  65],# 0)  Cervena -  nesplane budovy 
    [243, 149, 174],# 1)  Ruzova - vinohrady
    [237, 206,  65],# 2)  Zluta - spalne budovy
    [  0,   0,   0],# 3)  Cerna - hranice parcel, cisla parcel,popisy
    [109,  91,  92],# 4)  Tmavohneda - lesy
    [242, 216, 193],# 5)  Okrova - pole
    [ 49, 131, 103],# 6)  Syta zelena - parky, zahrady
    [133, 200, 155],# 7)  Zelena - louky
    [192, 224, 187],# 8)  Svetla zelena - pastviny
    [243, 248, 245],# 9)  Bila - verejna prost., neobd. puda, nadvori
    [196, 176, 165],# 10) Hneda - cesty
    [187, 234, 254]#  11) Modra - vodni plochy 
])
# order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# order = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# reorder = [x for _, x in sorted(zip(order, co_pal), key=lambda pair: pair[0])]
# new_pal = np.array(reorder)
test_pal_1 = np.array([
    [242, 216, 193],# 5)  Okrova - pole
    [243, 248, 245],# 9)  Bila - verejna prost., neobd. puda, nadvori
    [109,  91,  92],# 4)  Tmavohneda - lesy
    [187, 234, 254],#  11) Modra - vodni plochy
    [ 49, 131, 103],# 6)  Syta zelena - parky, zahrady
    [188,  14,  65],# 0)  Cervena -  nesplane budovy 
    [192, 224, 187],# 8)  Svetla zelena - pastviny
    [188,  14,  65],# 0)  Cervena -  nesplane budovy 
    [243, 149, 174],# 1)  Ruzova - vinohrady
    [237, 206,  65],# 2)  Zluta - spalne budovy
    [133, 200, 155],# 7)  Zelena - louky,
    [196, 176, 165],# 10) Hneda - cesty
])

RESC_FACT = 30
#
# %% [markdown]
- - - 
# %% LISTING ALL IMAGES FROM BALANCED AND TAKES ONE
all_imgs = file_lister('cb_then_wb') 
f_im = im_resize(load_im(all_imgs[10]), RESC_FACT) # rescales it by factor
# %% LISTING ALL IMAGES FROM ORIGINAL AND TAKES ONE
oall_imgs = file_lister('samples') 
o_im = im_resize(load_im(oall_imgs[8]), RESC_FACT) # rescales it by factor
#%% CONVERT TO HSV image
hsv_im = rgb2hsv(f_im)
hsv_im_o = rgb2hsv(o_im)
# %% PERFORM CLUSTERING AND RETURN LABELS
''' CLUSTERING IS DEAD END '''
gm_tied_lbl = label_predict(f_im, 1, 12, 3, 'spherical')
# %% PERFORM CLUSTERING AND RETURN LABELS
hsv_gm_spher_lbl = label_predict(hsv_im, 1, 12, 3, 'sp')
# %% PERFORM CLUSTERING AND RETURN LABELS
bi_lbl = label_predict(hsv_im, 2, 12, 3) # NOT WORKING
# %%
gm_tied_lbl.shape
# %%
save_im('9_gm_spherical_hsv_2.jpg',gm_tied_lbl, test_pal_1)


# %%
all_imgs[10], oall_imgs[8]
# %%
fig = plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
ax1.imshow(f_im)
ax2.imshow(o_im)
# %%
'''POLAR PLOT OF HSV PALETTE AND IMAGE'''
def plot_polar_hsv(hsv_im, rgb_im, size=10):
    # HSV color palette around circle
    xval = np.arange(0, 2*np.pi, 0.01)
    yval = np.ones_like(xval)
    color_m = plt.get_cmap('hsv')
    norm = Normalize(0.0, 2*np.pi)

    # HSV image components reshape for X and Y axes
    hsv_im_H = (hsv_im[:, :, 0]*2*np.pi).reshape(-1,1)
    hsv_im_S = hsv_im[:, :, 1].reshape(-1,1)
    hsv_im_V = hsv_im[:, :, 2].reshape(-1,1)
    hsv_im_SV = np.sqrt(hsv_im_S*hsv_im_V)
    # Create RGBA palete of HSV img-pixels
    rgba = np.dstack((rgb_im, np.ones(rgb_im.shape[:-1])))
    colormap = rgba.reshape(-1,4)

    fig = plt.figure(figsize=(18,18))
    ax1 = fig.add_subplot(231, projection='polar')
    ax2 = fig.add_subplot(232, projection='polar')
    ax3 = fig.add_subplot(233, projection='polar')
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    ax1.scatter(hsv_im_H, hsv_im_S, c=colormap, s=size, alpha=0.75, edgecolors=None )
    ax1.scatter(xval, yval, c=xval, s=30, cmap=color_m, norm=norm, linewidths=0)
    ax2.scatter(hsv_im_H, hsv_im_V, c=colormap, s=size, alpha=0.75, edgecolors=None )
    ax2.scatter(xval, yval, c=xval, s=30, cmap=color_m, norm=norm, linewidths=0)
    ax3.scatter(hsv_im_H, hsv_im_SV, c=colormap, s=size, alpha=0.75, edgecolors=None )
    ax3.scatter(xval, yval, c=xval, s=30, cmap=color_m, norm=norm, linewidths=0)
    ax4.scatter(hsv_im_H, hsv_im_S, c=colormap, s=size, alpha=0.75, edgecolors=None )
    ax4.scatter(xval, yval, c=xval, s=30, cmap=color_m, norm=norm, linewidths=0)
    ax5.scatter(hsv_im_H, hsv_im_V, c=colormap, s=size, alpha=0.75, edgecolors=None )
    ax5.scatter(xval, yval, c=xval, s=30, cmap=color_m, norm=norm, linewidths=0)
    ax5.set_yticks([])
    ax6.scatter(hsv_im_H, hsv_im_SV, c=colormap, s=size, alpha=0.75, edgecolors=None )
    ax6.scatter(xval, yval, c=xval, s=30, cmap=color_m, norm=norm, linewidths=0)
    ax6.set_yticks([])
    plt.show()
# %%
plot_polar_hsv(hsv_im, f_im, 2)
# %%
plot_polar_hsv(hsv_im_o, o_im, 2)
# %%
'''
class sklearn.cluster.AgglomerativeClustering(n_clusters=2, *
affinity='euclidean', memory=None, connectivity=None
compute_full_tree='auto', linkage='ward',
distance_threshold=None, compute_distances=False)[source]¶


cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average')
distance_matrix = sim_affinity(X)
cluster.fit(distance_matrix)
'''
from sklearn.cluster import AgglomerativeClustering

X = hsv_im.reshape(-1,3)
cluster = AgglomerativeClustering(n_clusters=12, affinity='precomputed', linkage='average')