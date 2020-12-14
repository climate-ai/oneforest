import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys
#package = os.path.dirname(os.getcwd())
#sys.path.append(package)
from deepforest_detection import convert_lonlat_to_xy, convert_lonlat_to_xy_tile


def box_to_annotation(boxes):
    """ Convert boxes as numpy array into a dictionary with all labels as Tree

    # Arguments
        box (array)    : A list of 4 elements (x1, y1, x2, y2).
    # Output
        annotations (dict) : dictionary with keys (bboxes, labels) for drawing annotations
    """
    e = np.tile("Tree", boxes.shape[0])[None].T
    annotations = {'bboxes': boxes, 'labels': e}
    return(annotations)



def draw_box(image, box, color, thickness=1):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    image = np.array(image)
    cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thickness, cv2.LINE_AA)
    return(image)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    fontScale = 1 #image.shape[0]*image.shape[1]/(1000*1000)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, fontScale, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, fontScale, (255, 255, 255), 1)



def draw_boxes(image, boxes, color, thickness=1):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)



def draw_detections(image, boxes, scores, labels, color=(0, 0, 255), label_to_name=None, score_threshold=0.05):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)


        # draw labels
        #caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        #draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=None, show_caption = False, plot = True, cv2_authorized = True, thickness=1):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else (255,0,0)
        if show_caption:
          caption = '{}'.format(label_to_name(label) if label_to_name else label)
          draw_caption(image, annotations['bboxes'][i], caption)
        image = draw_box(image, annotations['bboxes'][i], color=c, thickness = thickness)
    if plot:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if cv2_authorized:
        cv2.imshow('image',image)
    return(image)

def center(lst):
    return((lst[0]+lst[1])/2)

def get_tile_size(pt_ground_x, pt_ground_y, Xmin, Ymin, Xmax, Ymax, d=200):
    x0 = min(pt_ground_x, Xmin) - d
    x1 = max(pt_ground_x, Xmax) + d
    y0 = min(pt_ground_y, Ymin) - d
    y1 = max(pt_ground_y, Ymax) + d

    lst = [[x0, x1], [y0, y1]]

    dim = [x1 - x0, y1 - y0]
    id = np.argmax(dim)
    max_dim = np.max(dim)

    D = np.zeros((2,2))
    D[id] = lst[id]
    D[1-id] = [center(lst[1-id])-max_dim/2, center(lst[1-id])+max_dim/2]
    return(D)

def plot_circle(im, pt_x, pt_y, radius=15, color=(0, 0, 255), thickness=-1):
    for i in range(len(pt_x)):
        cv2.circle(im, (pt_x[i], pt_y[i]), radius, color, thickness)
    return(im)

def visualize_site_data(site_name, final, annotations_files, ortho_data):

    boxes = annotations_files[["Xmin", "Ymin", "Xmax", "Ymax"]].to_numpy()
    bboxes = box_to_annotation(boxes)
    img_path = 'wwf_ecuador/RGB Orthomosaics/{}.tif'.format(site_name)
    
    plt.figure(figsize = (15,15))
    im = cv2.imread(img_path)
    im = draw_annotations(im, bboxes, color=(0, 0, 255), label_to_name=None, show_caption = True, cv2_authorized = False, thickness = 5)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    for index, row in final.iterrows():
        
        lon_drone, lat_drone = row.lon_d, row.lat_d
        pt_drone_x, pt_drone_y = row.X_d, row.Y_d

        lon_ground, lat_ground = row.lon_g, row.lat_g
        pt_ground_x, pt_ground_y = row.X_g, row.Y_g

        plt.scatter(x=pt_ground_x, y=pt_ground_y, c='b', s=60)
        plt.scatter(x=pt_drone_x, y=pt_drone_y, c='r', s=60)
        
        plt.plot([pt_ground_x, pt_drone_x], [pt_ground_y, pt_drone_y], color='black', marker=None, linewidth=0.5)

        #text = "Name: {}\n Year: {}\n Diameter: {}".format(row['name'], row['year'], round(row['diameter'],4))
        #ax.annotate(text, (pt_drone_x, pt_drone_y), textcoords="offset points", # how to position the text
                        #xytext=(0,10), # distance from text to points (x,y)
                        #ha='left',bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()

def visualise_single_prediction(site_name, index, final, ortho_data, centered = True):
    row = final.loc[index]

    Xmin, Ymin, Xmax, Ymax = row.Xmin, row.Ymin, row.Xmax, row.Ymax
    box = np.array([[Xmin, Ymin, Xmax, Ymax]])
    bbox = box_to_annotation(box)

    lon_drone, lat_drone = row.lon_d, row.lat_d
    pt_drone_x, pt_drone_y = row.X_d, row.Y_d

    lon_ground, lat_ground = row.lon_g, row.lat_g
    pt_ground_x, pt_ground_y = row.X_g, row.Y_g
    
    text = "Name: {}\n Year: {}\n Diameter: {}".format(row['name'], row['year'], round(row['diameter'],4))
    
    img_path = 'wwf_ecuador/RGB Orthomosaics/{}.tif'.format(site_name)

    fig, (ax1, ax2) = plt.subplots(1,2 ,figsize = (10,10))
    if centered == True:
        p = 1000 #tile size
        ax1.set(xlim= (pt_ground_x-p/2, pt_ground_x+p/2), ylim=(pt_ground_y-p/2, pt_ground_y+p/2))
        ax2.set(xlim= (pt_drone_x-p/2, pt_drone_x+p/2), ylim=(pt_drone_y-p/2, pt_drone_y+p/2))
    else:
        D = get_tile_size(pt_ground_x, pt_ground_y, Xmin, Ymin, Xmax, Ymax, d=200)
        ax1.set(xlim= (D[0][0], D[0][1]), ylim=(D[1][0], D[1][1]))
        ax2.set(xlim= (D[0][0], D[0][1]), ylim=(D[1][0], D[1][1]))
    

    im = cv2.imread(img_path)
    im = draw_annotations(im, bbox, color=(0, 0, 255), label_to_name=None, show_caption = True, cv2_authorized = False)
    
    ax1.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax1.scatter(x=pt_ground_x, y=pt_ground_y, c='b', s=60)
    ax1.scatter(x=pt_drone_x, y=pt_drone_y, c='r', s=60)

    ax2.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax2.scatter(x=pt_drone_x, y=pt_drone_y, c='r', s=60)
    ax2.scatter(x=pt_drone_x, y=pt_drone_y, c='b', s=60)

    ax2.annotate(text, (pt_drone_x, pt_drone_y), textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='left',bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()


def visualise_multiple_prediction(site_name, lst_index, final, ortho_data):
    img_path = 'wwf_ecuador/RGB Orthomosaics/{}.tif'.format(site_name)

    row = final.loc[lst_index]
    n_tile = len(lst_index)

    Xmin, Ymin, Xmax, Ymax = row.Xmin, row.Ymin, row.Xmax, row.Ymax
    boxes = row[['Xmin', 'Ymin', 'Xmax', 'Ymax']].to_numpy()
    tree_char = row[['name', 'year', 'diameter']].to_numpy()
    text = ['Name: {}\n Year: {}\n Diameter: {}'.format(tree_char[k][0], tree_char[k][1], round(tree_char[k][2],2)) for k in range(n_tile)]
    annotations = {'bboxes': boxes, 'labels': np.array(text)}

    lon_drone, lat_drone = row.lon_d, row.lat_d
    pt_drone_x, pt_drone_y = row.X_d.values.astype(int), row.Y_d.values.astype(int)

    lon_ground, lat_ground = row.lon_g, row.lat_g
    pt_ground_x, pt_ground_y = row.X_g.values.astype(int), row.Y_g.values.astype(int)
   
    p = 1000 #tile size

    fig, axes = plt.subplots(n_tile ,2 ,figsize = (10, n_tile*5))

    im = cv2.imread(img_path)
    im = draw_annotations(im, annotations, color=(0, 0, 255), label_to_name=None, show_caption = False, plot = False, cv2_authorized = False)
    im = plot_circle(im, pt_drone_x, pt_drone_y, color=(0, 0, 255))

    im_before = plot_circle(im, pt_ground_x, pt_ground_y, color=(255, 0, 0))
    im_after = plot_circle(im, pt_drone_x, pt_drone_y, color=(255, 0, 0))


    for k in range(n_tile):
        
        imcut0 = im_before[int(pt_ground_y[k]-p/2):int(pt_ground_y[k]+p/2), int(pt_ground_x[k]-p/2):int(pt_ground_x[k]+p/2)]
        axes[k][0].imshow(cv2.cvtColor(imcut0, cv2.COLOR_BGR2RGB))
        
        imcut1 = im_after[int(pt_drone_y[k]-p/2):int(pt_drone_y[k]+p/2), int(pt_drone_x[k]-p/2):int(pt_drone_x[k]+p/2)]
        axes[k][1].imshow(cv2.cvtColor(imcut1, cv2.COLOR_BGR2RGB))
        axes[k][1].annotate(text[k], (p/2, p/2), textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='left',bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()


def visualize_tile_prediction(site_name, tile, final, annotations_files, ortho_data, params):

    """
        params (dict): dictionary containing paramaters for the matching. 
            Keys: ot_params (position, is_musacea), ot_type (sinkhorn, emd), acc_musacea (accuracy for CNN prediction after matching), merge (greedy, non-greedy)
    """

    final_tile = final[final.img_path == tile].reset_index(drop=True)

    fig, ax = plt.subplots(1 ,figsize = (20,20))

    tile_xmin, tile_ymin = final_tile.loc[0].tile_xmin, final_tile.loc[0].tile_ymin

    # Plot all detected boxes by DeepForest on the selected tile
    annotations_tile = annotations_files[annotations_files.img_path == tile].reset_index(drop=True)
    boxes = annotations_tile[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
    bboxes = box_to_annotation(boxes)
    img_path = 'images/{}/{}'.format(site_name, tile)
    im = cv2.imread(img_path)
    im = draw_annotations(im, bboxes, color=(0, 0, 255), label_to_name=None, plot = False, cv2_authorized = False, thickness = 4)

    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    #ax.set(xlim= (0, 4000), ylim=(0, 4000))

    lst = [value for key, value in params.items()]
    txt = 'Params: {} - Matching: {} - Musacea accuracy = {} - Merge: {}'.format(lst[0], lst[1], lst[2], lst[3])
    plt.figtext(0.5, 0.085, txt, wrap=True, horizontalalignment='center', fontsize=16)


    for index, row in final_tile.iterrows():
        
        lon_drone, lat_drone = row.lon_d, row.lat_d
        pt_drone_x, pt_drone_y = row.x, row.y

        lon_ground, lat_ground = row.lon_g, row.lat_g
        pt_ground_x, pt_ground_y = row.X_g - tile_xmin, row.Y_g - tile_ymin

        ax.scatter(x=pt_ground_x, y=pt_ground_y, c='b', s=60)
        ax.scatter(x=pt_drone_x, y=pt_drone_y, c='r', s=60)
        
        ax.plot([pt_ground_x, pt_drone_x], [pt_ground_y, pt_drone_y], color='black', marker=None, linewidth=0.5)

        #text = "Name: {}\n Year: {}\n Diameter: {}".format(row['name'], row['year'], round(row['diameter'],4))
        #ax.annotate(text, (pt_drone_x, pt_drone_y), textcoords="offset points", # how to position the text
                        #xytext=(0,10), # distance from text to points (x,y)
                        #ha='left',bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.show()