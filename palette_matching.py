# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib import gridspec
from scipy.spatial import cKDTree
import scipy.spatial.distance
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage.color import rgb2lab
from skimage.color import lab2rgb
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import numpy as np
import cv2
from PIL import Image
import os
import itertools as it

# %% [markdown]
# - - - 
# ##  Přebarvení rastrů CO podle předem definované palety barev  
#  - pomocí scipy.spatial.cKDTree algoritmu [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html] 
#  - test proveden v barevném prostoru RGB, HSV a LAB(CIELAB)  
#  
#  **Paleta** (viz.: Obtain color-palette using clustering with k-means algorithm from **"KmeansClustering.ipynb"**):  
#  ![24color paleta](rgb24_pal.png "Použitá paleta")
# %% [markdown]
#  - - - 
#  ### Helper functions

# %%
def img_resize(image, n=480):
    '''Rescale image to default size of n
    
    Maintains the aspect ratio rescales the longer side to required n
    
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


# %%
def visualize_palette(palette, scale=0):
    '''Visualizes a palette as a rectangle of increasingly "bright" colors.

    This method first converts the RGB pixels into grayscale and sorts the
    grayscale pixel intensity as a proxy of sorting the RGB pixels. Then the
    pixels are reshaped into a 2:1 rectangle and displayed. If there are more
    fewer pixels tahn the size of the rectangle, the remaining pixels are given
    a generic gray color.

    Args
    ---
        palette (numpy.ndarray): the RGB pixels of a  color palette.
        scale (int): the scale factor to apply to the image of the palette.
    '''
    palette_gray = palette @ np.array([[0.21, 0.71, 0.07]]).T
    idx = palette_gray.flatten().argsort()
    h, w = closest_rect(palette.shape[0])
    palette_sorted = palette[idx]
    padding = (h*w) - palette.shape[0]
    
    if (h*w) > palette.shape[0]:
        palette_sorted = np.vstack(
            (palette_sorted, 51*np.ones((padding, 3), dtype=np.uint8))
        )

    palette_sorted = palette_sorted.reshape(h, w, 3)
    im = Image.fromarray(palette_sorted)
    
    if scale > 0:
        return im.resize((scale*im.width, scale*im.height), Image.NEAREST)

    return im


# %%
def closest_rect(n):
    '''Finds the closest height and width of a 2:1 rectangle given the number
    of pixels.

    Args
    ---
        n (int): the number of pixels in an image.
    '''
    k = 0
    while 2 * k ** 2 < n:
        k += 1
    
    return k, 2*k


# %%
# OpenCV method to resize IMAGE-ARRAY
# we use resized array to shorten plot time
def res_img_arr(img, x=240, y=180):
    '''
    Uz se mi ani neche psat docstring
    '''
    res_img = cv2.resize(img, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    return res_img


# %%
def plot_samples(pics, rows=3,cols=4, offset=0, fig_x=16, fig_y=9, im_rsz=320):
    plt.figure()
    offset = offset
    rows, cols = rows, cols
    f, ax = plt.subplots(rows, cols,figsize=(fig_x,fig_y)) 
    count = 0
    for row in range(rows):
        for col in range(cols):
            pic_name = f'{pics[count+offset]}'
            img = img_resize(Image.open(os.path.join(samples, pic_name)),im_rsz)
            ax[row, col].imshow(img)
            ax[row, col].set_yticklabels([])
            ax[row, col].set_xticklabels([])
            ax[row, col].annotate(f'{pic_name} \nIndex: {count}',
                                  (0.1, 0.5), xycoords='axes fraction', va='center')
            count += 1

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()


# %%
class multiArrayPicture:
    '''
    TODO docstring
    '''
    def __init__(self, img):
        self.img = img
        
    def img_samples(self, samp):
        self.samp = samp
        


# %%
# converting above RGB picture to float HSV values  ..as numpy array of course
def toHSV(img):
    img = rgb2hsv(img)
    return img

def fromHSV(img):
    img = img_as_ubyte(hsv2rgb(img))
    return img


# %%
def toLAB(img):
    img = rgb2lab(img)
    return img

def fromLAB(img):
    img = img_as_ubyte(lab2rgb(img))
    return img


# %%
def arrLookup(arr):
    print(arr.shape, arr.dtype)
    print(arr[:2,:3,:])


# %%
def imgAsRescaledArray(idx, folder, x=400, y=300):
    samples = os.path.join(os.getcwd(), folder)
    pics = os.listdir(samples)
    img = Image.open(os.path.join(samples, pics[idx]))
    img = np.array(img)
    img = res_img_arr(img, x, y)
    return img

# %% [markdown]
#  - - - 
#  ### List and display sample images

# %%
samples = os.path.join(os.getcwd(), 'samples')
pics = os.listdir(samples)
pics


# %%
plot_samples(pics)


# %%
# Prepare image for analysis - downsample and convert to array
img1 = imgAsRescaledArray(4, 'samples', 900, 600)


# %%
arrLookup(img1)

# %% [markdown]
# - - - 
# ### Konverze RGB to HSV

# %%
# RGB image index 1 to HSV
img1_hsv = toHSV(img1)


# %%
# View array composition
arrLookup(img1_hsv)

# %% [markdown]
# - - - 
# ### Konverze RGB to CIELAB
#     **Function name**    **Description**
#     
#     img_as_float    Convert to 64-bit floating point.
#     img_as_ubyte    Convert to 8-bit uint.
#     img_as_uint    Convert to 16-bit uint.
#     img_as_int    Convert to 16-bit int.
# 
# img_as_ubyte(image)

# %%
# RGB image index 1 to LAB
img1_lab = toLAB(img1)


# %%
# View array composition
arrLookup(img1_lab)

# %% [markdown]
# - - - 
# ### Definovani vlastni palety barev

# %%
# testovaci k=24 color paleta vytvorena k-mean clustering
# na jednom zvolenem optimal. obrazku
rgb_24_color_palette = np.array([[167, 199, 147],
       [247, 224, 188],
       [135, 133,  99],
       [206, 217, 169],
       [246, 235, 206],
       [ 48,  45,  21],
       [178, 210, 201],
       [196, 182, 156],
       [111, 104,  72],
       [248, 187, 142],
       [186, 172, 147],
       [193, 213, 161],
       [250, 176, 179],
       [ 52, 136, 160],
       [159, 158, 122],
       [216, 196, 160],
       [217, 219, 175],
       [244, 225,  90],
       [ 79,  74,  49],
       [242, 230, 199],
       [246, 219, 180],
       [185, 106,  79],
       [191, 216, 204],
       [181, 208, 155]]).astype(np.uint8)
visualize_palette(rgb_24_color_palette, scale=32)


# %%
def toFloat(img):
    img = img_as_float(img)
    return img

def fromFloat(img):
    img = img_as_ubyte(img)
    return img


# %%
# normovana RGB paleta [0 az 1] jako list pro 3D plot barevnych prostoru
# normRGBpalette = img_as_float(rgb_24_color_palette).tolist()

# %% [markdown]
# - - - 
# ## OpenCV testing 3D plotinng of colourspace

# %%
# PLOTTING 3D COLOURSPACE OF PALETTE IN RGB


# %%
pal = cv2.imread('rgb24_pal.png')
plt.imshow(pal)
plt.show()


# %%
pal = cv2.cvtColor(pal, cv2.COLOR_BGR2RGB)
plt.imshow(pal)
plt.show()


# %%
# PLOTTING USING RGB 24-COLOR-PALETTE FROM ARRAY

# prepare RGB image
res_rgb = res_img_arr(pic11_arr)
# split image array
r, g, b = cv2.split(res_rgb)

# prepare color palette plotting coordinates
rp = rgb_24_color_palette[:,0]
gp = rgb_24_color_palette[:,1]
bp = rgb_24_color_palette[:,2]

fig = plt.figure(figsize=(12,9))
# fig = plt.figure(figsize=plt.figaspect(0.5))

# Plot figure A
axis = fig.add_subplot(1, 2, 1, projection="3d")
axis.title.set_text('RGB color-space')
## IMPORTANT ## color preparation  for colouring other spaces .. HSV, LAB ##
pixel_colors = res_rgb.reshape((np.shape(res_rgb)[0]*np.shape(res_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist() # final color palete product 'pixel_colors' for further plotting
##
# Image array subplot
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".", s=150)
# Palette subplot
axis.scatter(rp, gp, bp, facecolors=normRGBpalette, marker=".", edgecolors= "cyan", s=850)
# Axis labels
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
axis.view_init(30,-40)

# Plot figure B
axis = fig.add_subplot(1, 2, 2, projection="3d")
# Image array subplot
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".", s=150)
# Palette subplot
axis.scatter(rp, gp, bp, facecolors=normRGBpalette, marker=".", edgecolors= "cyan", s=850)

# Axis labels
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
axis.view_init(30,180)

plt.show()


# %%
# konverze RGB palety do HSV
hsv_24_color_palette = rgb2hsv(rgb_24_color_palette).astype(np.float32)

# %% [markdown]
# - - - 
# ## 3D plot function

# %%
def plotQuat3D(img, pal, img_rgb, pal_rgb, xview=30, yview=[-45, 45, 120 ,200] ):
    # prepare image
    x, y, z = cv2.split(img)
    # prepare color palette
    pal_x = pal[:,0]
    pal_y = pal[:,1]
    pal_z = pal[:,2]
    
    img_colors = toFloat(img_rgb.reshape(-1, img_rgb.shape[-1])).tolist()
    pal_colors = toFloat(pal_rgb.reshape(-1, pal_rgb.shape[-1])).tolist()
    
    nrow = 2
    ncol = 2
    
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.0, hspace=0.0)
    
    fig = plt.figure(figsize=(18,14))
    
    count = 0
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot2grid((nrow,ncol), (i, j), projection="3d")
            ax.scatter(x.flatten(), y.flatten(), z.flatten(), facecolors=img_colors, marker=".", s=1, zorder=-1)
            ax.scatter(pal_x, pal_y, pal_z, facecolors=pal_colors, marker=".", edgecolors= "blue", s=450, zorder=2)
#             ax.set_yticklabels([])
#             ax.set_xticklabels([])
#             ax.set_zticklabels([])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(xview,yview[count])
            count += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# %%
plotQuat3D(img1, rgb_24_color_palette, img1, rgb_24_color_palette)


# %%
plotQuat3D(toHSV(img1), toHSV(rgb_24_color_palette), img1, rgb_24_color_palette)


# %%
plotQuat3D(toLAB(img1), toLAB(rgb_24_color_palette), img1, rgb_24_color_palette)

# %% [markdown]
# - - - 
# ### Reklasifikace pixelu obrazku metodou "nearest-neighbor lookup"

# %%
def ckDTreeReclass(img, pal):
    recl_img = pal = pal[cKDTree(pal).query(img,k=1)[1]]
    return recl_img


# %%
get_ipython().run_cell_magic('time', '', '# RGB klasifikace podle palety\nimg1_recl_rgb = ckDTreeReclass(img1, rgb_24_color_palette)')


# %%
get_ipython().run_cell_magic('time', '', '# HSV klasifikace podle palety a prevod na RGB integer\nimg1_recl_hsv = ckDTreeReclass(toHSV(img1), toHSV(rgb_24_color_palette))\nimg1_recl_hsv = fromHSV(img1_recl_hsv)')


# %%
get_ipython().run_cell_magic('time', '', '# LAB klasifikace podle palety\nimg1_recl_lab = ckDTreeReclass(toLAB(img1), toLAB(rgb_24_color_palette))\nimg1_recl_lab = fromLAB(img1_recl_lab)')


# %%
recl_imgs = [img1, img1_recl_rgb, img1_recl_hsv, img1_recl_lab]

for i in recl_imgs:
#     print(f'Reclassified array of: {i}')
    print(i.shape, i.dtype)
    print(i[:9,4,:])
    print('\n')


# %%
# DISPLAYING OUR RECLASIFF IMAGES
# RGB
display(Image.fromarray(img1_recl_rgb))


# %%
# HSV
display(Image.fromarray(img1_recl_hsv))


# %%
# LAB
display(Image.fromarray(img1_recl_lab))


# %%
recl_names = ['img1','img1_recl_rgb', 'img1_recl_hsv', 'img1_recl_lab']
counter = 0
for img in recl_imgs:
    Image.fromarray(img).save(f'{recl_names[counter]}_{pics[1]}')
    print(f'{recl_names[counter]}_{pics[1]}')
    counter += 1


# %%
res = [x for x in os.listdir(os.getcwd()) if x.endswith('5.jpg')]
res


# %%
display('img1_recl_rgb_0347-1-005.jpg')


# %%
get_ipython().run_cell_magic('time', '', "plt.figure()\n#subplot(r,c) provide the no. of rows and columns\n\noffset = 0\n\nrows, cols = 2, 2\n\nf, ax = plt.subplots(rows, cols,figsize=(24,18)) \n\ncount = 0\nfor row in range(rows):\n    for col in range(cols):\n        pic_name = f'{res[count+offset]}'\n        img = Image.open(pic_name)\n        ax[row, col].imshow(img)\n        ax[row, col].set_yticklabels([])\n        ax[row, col].set_xticklabels([])\n        ax[row, col].annotate(f'{pic_name} \\nIndex: {count}',\n                              (0.1, 0.5), xycoords='axes fraction', va='center')\n        count += 1\n\nplt.subplots_adjust(wspace=0, hspace=0.1)\nplt.show()")

# %% [markdown]
# ### TODO:
#     1. plotting palette as plt.patches with RGB codes
#     2. do not apply PALETTE using euclidean spatial distance
#     3. instead implement mahalanabis(covariance) non-euclidean spatial distance
#     4. refactoring code to set of callable functions or OOPclasses 
#       
# %% [markdown]
# -  - -
# ## Computing spatial distance

# %%
palette  = np.array([[0, 0, 0], 
[0, 0, 255], 
[255, 0, 0], 
[150, 30, 150], 
[255, 65, 255], 
[150, 80, 0], 
[170, 120, 65], 
[125, 125, 125], 
[255, 255, 0], 
[0, 255, 255], 
[255, 150, 0], 
[255, 225, 120], 
[255, 125, 125], 
[200, 100, 100], 
[0, 255, 0], 
[0, 150, 80], 
[215, 175, 125], 
[220, 180, 210], 
[125, 125, 255]
])

valueRange = np.arange(0,256)
allColors = np.array(list(it.product(valueRange,valueRange,valueRange)))
mapping = scipy.spatial.distance.cdist(allColors, palette).argmin(1)


