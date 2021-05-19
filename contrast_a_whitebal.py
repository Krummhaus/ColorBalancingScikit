# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
import os
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL import ImageColor
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from pathlib import Path
from scipy.spatial import distance
import scipy
from skimage import io

# %%
def plot_samples(folder, save=False, rows=3,cols=4, offset=0, fig_x=16, fig_y=9, im_rsz=300):
    plt.figure()
    offset = offset
    rows, cols = rows, cols
    f, ax = plt.subplots(rows, cols,figsize=(fig_x,fig_y)) 
    # get path as path-like-object cause of Win/Unix
    basepath = Path(os.getcwd())
    # list of jpg-s in folders as path-like-object
    pics_list= list(basepath.glob(f'{folder}/*.jpg'))
    count = 0
    for row in range(rows):
        for col in range(cols):
            subdir = os.path.split(pics_list[count+offset])[0].split(os.sep)[-1]
            picture = os.path.split(pics_list[count+offset])[1]
            img = img_resize(Image.open(pics_list[count+offset]),im_rsz)
            ax[row, col].imshow(img)
            ax[row, col].set_yticklabels([])
            ax[row, col].set_xticklabels([])
            ax[row, col].annotate(f'{picture} \nIndex: {count}',
                                  (0.1, 0.5), xycoords='axes fraction', va='center')
            count += 1

    plt.subplots_adjust(wspace=0, hspace=0.1)

    if save == True:
        plt.savefig(f'{os.path.join(os.getcwd(), subdir, f"SET_{subdir}.jpg")}', dpi=200)
        print(f'{os.path.join(os.getcwd(), subdir, f"SET_{subdir}.jpg")}')

    plt.show()


def show_pal(pal):
    fig, ax = plt.subplots()
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    idxs = np.arange(len(pal)).reshape(3,int(len(pal)/3))
    io.imshow(pal[idxs])
    counter = 0
    for i in range(3):
        for j in range(int(len(pal)/3)):
            ax.annotate(pal.tolist()[counter],(j,i), ha='center')
            counter += 1


def file_lister(folder):
        basepath = Path(os.getcwd())
        pics_list= list(basepath.glob(f'{folder}/*.jpg'))
        return pics_list


def open_img_as_arr(img):
    img = np.asarray(Image.open(img))
    return img
    
def toFloat(img):
    img = img_as_float(img)
    return img

def fromFloat(img):
    img = img_as_ubyte(img)
    return img


def res_img_arr(img, x=240, y=180):
    '''Resize imagae using OpenCV

    without maintainig aspect ratio, default with CUBIC interpolation

    Args
    ---
        img : image opend as numpy array
        x, y : output size of image
        TODO: if needed we can implement interpolation method, not needed atm.
    '''
    res_img = cv2.resize(img, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    return res_img


def img_resize(image, n=800):
    '''Resize image to default size of n maintaining aspect ratio using PIL
    
    maintains the aspect ratio rescales the longer side to required n
    
    Args
    ---
        image (Image.fromarray): RGB image from numpy array with PIL Image.fromarray()
        n (int): required size of image, default is n
    '''
    if image.width > n or image.height > 1028:
        if image.height > image.width:
            factor = n / image.height
        else:
            factor = n / image.width
        tn_image = image.resize((int(image.width * factor), int(image.height * factor)))
        return tn_image


def plotQuat3D(img, pal, img_rgb, pal_rgb, nrow=2, ncol =2, xview=30):
    '''Plots image color space on x,y,z axes using matplotlib
    
    takes intput image array and plot each respective pixel as point in 3D space
    whit its apropriate color.

    Args
    ---
        img: image array as RGB, HSV, LAB etc. array. Can be UINT8 or FLOAT array
        pal: color palette with same dtype as above 'img'
        img_rgb: same image as above'img' but with RGB color codes, it is internaly
                 rescaled to FLOAT RGB color in interval (0, 1) -> couse of matplotlib
        img_pal: same palette as above 'pal' but with RGB color codes, it is internaly
                 rescaled to FLOAT RGB color in interval (0, 1) -> couse of matplotlib
        nrow, ncol: number of rows and columns with rotated 3D subplots
        xview: number(degrees), rotation of 3D subplots by X axis, default is 30
    '''
    # prepare rotations
    def y_view(nrow, ncol, sta=-60, stp=210):
        y_vw_list = []
        for i in range(sta, stp, int(270/(nrow*ncol))):
            y_vw_list.append(i)
        return y_vw_list

    yview=y_view(nrow, ncol, sta=-60, stp=210)
    
    # prepare image
    x, y, z = cv2.split(img)
    # prepare color palette
    pal_x = pal[:,0]
    pal_y = pal[:,1]
    pal_z = pal[:,2]
    
    img_colors = toFloat(img_rgb.reshape(-1, img_rgb.shape[-1])).tolist()
    pal_colors = toFloat(pal_rgb.reshape(-1, pal_rgb.shape[-1])).tolist()
    
    fig = plt.figure(figsize=(18,14))
    
    count = 0
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot2grid((nrow,ncol), (i, j), projection="3d")
            ax.scatter(x.flatten(), y.flatten(), z.flatten(), facecolors=img_colors, marker=".", s=1, zorder=-1)
            ax.scatter(pal_x, pal_y, pal_z, facecolors=pal_colors, marker=".", edgecolors= "blue", s=450, zorder=2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(xview,yview[count])
            count += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def reshape3Dto2D(img):
    img = img.reshape(-1, img.shape[-1])
    return img

def reshape2Dto3D(img, orig_img):
    img = img.reshape(orig_img.shape[0], orig_img.shape[1], 3).astype(np.uint8)
    return img

def mapp_to_palette(img, pal, typ_dist):
    '''
    typ_dist = ['euclidean', 'mahalanobis']
    '''
    x, y, z = img.shape[0], img.shape[1], 3
    mapping = scipy.spatial.distance.cdist(reshape3Dto2D(img), pal, metric=typ_dist).argmin(1)
    img = pal[mapping]
    img = img.reshape(x, y, z).astype(np.uint8)
    return img

def hex2rgb_arr(hex_list):
    rgb_list = [(ImageColor.getcolor(x, 'RGB')) for x in hex_list]
    rgb_arr = np.array(rgb_list).astype(np.uint8)
    return rgb_arr

# %%
plot_samples('samples', False)
# %%
plot_samples('cb_then_wb', False)
# %%
'''
Definice barev dle CO katastru

# Cervena -  nesplane budovy
# Ruzova - vinohrady
# Zluta - spalne budovy
# Cerna - hranice parcel, cisla parcel,popisy

# Tmavohneda - lesy
# Okrova - pole
# Syta zelena - parky, zahrady
# Zelena - louky

# Svetla zelena - pastviny
# Bila - verejna prost., neobd. puda, nadvori
# Hneda - cesty
# Modra - vodni plochy
'''

co_hex = ['#ff62a0', '#f68b81', '#ffe63c', '#06000e', '#535c61', '#efcab3',
        '#5e643c', '#a3a357', '#d4d9b9', '#edeee9', '#d5bab3', '#b0e6f7']


co_1_pal = np.array(hex2rgb_arr(co_hex)).astype(np.uint8)

# DALSI PALETY
co_2_pal_hex = '''\
#bc0e41 #f395ae #edce41 #000000 \
#6d5b5c #f2d8c1 #318367 #85c89b \
#c0e0bb #f3f8f5 #c4b0a5 #bbeafe'''.split(' ')
co_2_pal = hex2rgb_arr(co_2_pal_hex)

# %%
show_pal(co_2_pal)
#Testovaci palety
# %%

# %%
'''Load image as numpy array and scales it down for easier
computing'''
im = res_img_arr(open_img_as_arr(file_lister('cb_then_wb')[4]), 1024, 900)
# %%
'''Load image as numpy array NO DOWNSCALE'''
im_orig = open_img_as_arr(file_lister('cb_then_wb')[5])
# %%
im_raw = res_img_arr(open_img_as_arr(file_lister('samples')[4]), 200, 180)
# %%
'''Plotting image with another rescaling'''
# imr = res_img_arr(im, 200, 180)
plotQuat3D(res_img_arr(im, 200, 180), co_pal, res_img_arr(im, 200, 180), co_pal, 3, 4)
# %%
plotQuat3D(im_raw, rgb_pal, im_raw, rgb_pal, 3, 3)
# %%
'''Remaps all pixels of input image to color-palette
using euclidean distance'''
im_mapp_eucl = mapp_to_palette(im_orig, co_2_pal, 'euclidean')
# %%
'''Remaps all pixels of input image to color-palette
using mahalinobis distance'''
im_mapp_maha = mapp_to_palette(im_orig, co_2_pal, 'mahalanobis')
# %%
display(Image.fromarray(im_mapp_eucl))
# %%
display(Image.fromarray(im_mapp_maha))
# %%
Image.fromarray(im_mapp_eucl).save('co_pal_mapp_euclid.jpg')
# %%
Image.fromarray(im_mapp_maha).save('co_pal_mapp_maha.jpg')
# %%
# %%
# %%
im_idx4 = open_img_as_arr(file_lister('cb_then_wb')[4])

# %%
im_idx4_remap = mapp_to_palette(im_idx4, rgb_pal, 'euclidean')
# %%
def save_img(img, name):
    Image.fromarray(img).save(f'{name}.jpg')

# %%
save_img(im_mapp_eucl, 'co_pal_mapp_eucl_RGB')

# %%
show_pal(co_pal)

