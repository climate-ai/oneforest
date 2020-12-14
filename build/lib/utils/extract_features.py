import os

import pandas as pd
import numpy as np
import lxml.etree as etree

import exifread
from shapely.geometry import Polygon
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
from PIL import Image
import slidingwindow


# Read .kml files for Orthomosaics RGB

def read_kml(dir, file):
    kml_file = os.path.join(dir,file)
    x = etree.parse(kml_file)
    name = file.replace('.kml','')
    for el in x.iter(tag = "{*}south"):
        lat_min = float(el.text)
    for el in x.iter(tag = "{*}north"):
        lat_max = float(el.text)
    for el in x.iter(tag = "{*}west"):
        lon_min = float(el.text)
    for el in x.iter(tag = "{*}east"):
        lon_max = float(el.text)

    return([name, lat_min, lat_max, lon_min, lon_max])

def read_orthomosaics(dir):
    data = []
    for file in os.listdir(dir):
        if file.endswith('.kml'):
            data.append(read_kml(dir, file))
    return(pd.DataFrame(data = data, columns=['name', 'lat_min', 'lat_max', 'lon_min', 'lon_max']))


# Split Orthomosaics in 4000x4000 tiles

def image_name_from_path(image_path):
    """Convert path to image name for use in indexing."""
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    return image_name


def compute_windows(numpy_image, patch_size, patch_overlap):
    """Create a sliding window object from a raster tile.

    Args:
        numpy_image (array): Raster object as numpy array to cut into crops

    Returns:
        windows (list): a sliding windows object
    """

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return (windows)



def save_crop(base_dir, image_name, index, tile_position, crop):
    """Save window crop as image file to be read by PIL.

    Filename should match the image_name + window index
    """
    # create dir if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    im = Image.fromarray(crop)
    image_basename = os.path.splitext(image_name)[0]
    x0, y0, x1, y1 = tile_position
    filename = "{}/{}_{}_{}_{}_{}_{}.png".format(base_dir, image_basename, index, x0, y0, x1, y1)
    im.save(filename)

    return filename


def split_raster(path_to_raster,
                 base_dir="images",
                 patch_size=400,
                 patch_overlap=0.05):
    """Divide a large tile into smaller arrays. Each crop will be saved to
    file.

    Args:
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        base_dir (str): Where to save the annotations and image
            crops relative to current working dir
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1

    """
    # Load raster as image
    raster = Image.open(path_to_raster)
    numpy_image = np.array(raster)
    numpy_image = numpy_image[:, :, :3]

    # Check that its 3 band
    bands = numpy_image.shape[2]
    if not bands == 3:
        raise IOError("Input file {} has {} bands. DeepForest only accepts 3 band RGB "
                      "rasters in the order (height, width, channels). "
                      "If the image was cropped and saved as a .jpg, "
                      "please ensure that no alpha channel was used.".format(
                          path_to_raster, bands))

    # Check that patch size is greater than image size
    height = numpy_image.shape[0]
    width = numpy_image.shape[1]
    print('height', height)
    print('width', width)
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = compute_windows(numpy_image, patch_size, patch_overlap)

    # Get image name for indexing
    image_name = os.path.basename(path_to_raster)

    for index, window in enumerate(windows):

        #Crop image
        tile_pos = change_tile_xy_origin(windows[index].indices(), height)
        crop = numpy_image[windows[index].indices()]
        image_path = save_crop(base_dir, image_name, index, tile_pos, crop)

    return

def change_tile_xy_origin(win, Y):
    """
    Convert the window position to the x,y pixel position of the tile on the image
    The window position take origin on the top-left of the image
    The x,y position takes origin on the down-left corner of the image
    """
    x_min = win[1].start
    x_max = win[1].stop
    y_min = Y - win[0].stop
    y_max = Y - win[0].start
    return(x_min, y_min, x_max, y_max)


# Extract Orthomosaics features

def ratio(size, min, max):
    delta = max - min
    r = delta/float(size)
    return(r)


def create_ortho_data(directory, save_dir):
    ortho_features = read_orthomosaics(directory)

    ortho_dim = []
    for file in os.listdir(directory):
        if file.endswith('.tif'):
            # Open image file for reading (binary mode)
            path_to_raster = os.path.join(directory, file)
            f = open(path_to_raster, 'rb')

            # Return Exif tags
            tags = exifread.process_file(f)
            width = int(str(tags['Image ImageWidth']))
            height = int(str(tags['Image ImageLength']))
            name = file.replace('.tif','')
            ortho_dim.append([name, width, height])
    ortho_dim = pd.DataFrame(data = ortho_dim, columns=['name', 'width', 'height']) 
    
    ortho_data = pd.merge(ortho_features, ortho_dim, on = 'name')  
    ortho_data['ratio_x'] = ortho_data.apply(lambda x: ratio(x.width, x.lon_min, x.lon_max), axis=1)
    ortho_data['ratio_y'] = ortho_data.apply(lambda x: ratio(x.height, x.lat_min, x.lat_max), axis=1)
    ortho_data.to_csv(save_dir, index = False)
    return(ortho_data)



if __name__ == '__main__':
    directory = "wwf_ecuador/RGB Orthomosaics"
    save_dir = 'images'

    ortho_data = create_ortho_data(directory, save_dir)

    # Split images into tiles
    for file in os.listdir(directory):
        if file.endswith('.tif'):
            # Open image file for reading (binary mode)
            path_to_raster = os.path.join(directory, file)
            name = file.replace('.tif','')
            
            tiles_dir = os.path.join(save_dir,name)
            if not os.path.exists(tiles_dir):
                os.makedirs(tiles_dir)

            split_raster(path_to_raster, base_dir=tiles_dir, patch_size=4000, patch_overlap=0.05)