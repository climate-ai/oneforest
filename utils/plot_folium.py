import numpy as np
import datetime
import folium
from folium.map import *
from folium import plugins
from folium.plugins import MeasureControl
from folium.plugins import FloatImage

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

def line_coordinates(points_g0, points_g1, G):
    """
    Takes two lists of points coordinates and draw the lines connecting points according to the matrix M, with line width corresponding to the coefficient M_i,j
    Args:
        points_g0, points_g0 (array): list of 2D GPS coordinates of points
        G (matrix) : adjacency matrix (computed from the optimal transport plan)
    Output:
        connected_pairs (array): the list of pairs of connected points
        width (array): list of the width for each connection
    """
    idx_0,idx_1 = np.where(G!=0)
    connected_pairs = []
    width = []
    for k in range(len(idx_0)):
        coord_0 = points_g0[idx_0[k]]
        coord_1 = points_g1[idx_1[k]]
        connected_pairs.append([coord_0.tolist(), coord_1.tolist()])
        width.append(G[idx_0[k], idx_1[k]])
    return(connected_pairs, width)


def plot_initial(COORDINATES, X_aerial, X_street, df_street, satellite = True):
    # initialize empty map zoomed in on 
    m = folium.Map(location=COORDINATES, zoom_start=13,tiles='CartoDBPositron')

    #add markers
    for i in range(len(X_aerial)):
        #folium.Marker(each, icon=folium.Icon(color='blue', icon='cloud')).add_to(m)
        folium.CircleMarker(location=X_aerial[i], radius=3, weight = 0, fill_color='red', fill_opacity=0.4).add_to(m)

    for i in range(len(X_street)):
        #folium.Marker(each, icon=folium.Icon(color='blue', icon='cloud')).add_to(m)
        my_string = '<b>Name</b>: {}, <br><b>Year</b>: {}, <br><b>Tree_id</b>: {}, <br><b>Plot_id</b>: {},  <br><b>Diameter</b>: {},  <br><b>Height</b>: {}'.format(df_street.name[i], df_street.year[i], df_street.tree_id[i], df_street.plot_id[i], df_street.diameter[i], df_street.height[i])  
        folium.CircleMarker(location=X_street[i], radius=3, weight = 0, fill_color='blue', fill_opacity=0.4, popup = Popup(my_string)).add_to(m)

    if satellite:
        folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Esri Satellite',
                overlay = False,
                control = True
            ).add_to(m)


    # Draw Polygon
    shapefile = gpd.read_file("wwf_ecuador/Merged_final_plots/Merged_final_plots.shp")
    merged_final = pd.DataFrame(shapefile)
    for k in range(6):
        folium.GeoJson(merged_final.geometry.iloc[k]).add_to(m)

    return(m)


def plot_mapping(COORDINATES, X_aerial, X_street, G, df_street, satellite = True):
    # initialize empty map zoomed in on 
    m = folium.Map(location=COORDINATES, zoom_start=13,tiles='CartoDBPositron')

    #add markers
    for i in range(len(X_aerial)):
        #folium.Marker(each, icon=folium.Icon(color='blue', icon='cloud')).add_to(m)
        folium.CircleMarker(location=X_aerial[i], radius=5, weight = 0, fill_color='red', fill_opacity=0.4).add_to(m)

    for i in range(len(X_street)):
        #folium.Marker(each, icon=folium.Icon(color='blue', icon='cloud')).add_to(m)
        my_string = '<b>Name</b>: {}, <br><b>Year</b>: {}, <br><b>Tree_id</b>: {}, <br><b>Plot_id</b>: {},  <br><b>Diameter</b>: {},  <br><b>Height</b>: {}'.format(df_street.name[i], df_street.year[i], df_street.tree_id[i], df_street.plot_id[i], df_street.diameter[i], df_street.height[i])  
        folium.CircleMarker(location=X_street[i], radius=5, weight = 0, fill_color='blue', fill_opacity=0.4, popup = Popup(my_string)).add_to(m)

    ##add lines: connect two nodes if the predicted mapping coefficient is equal to 1
    connected_pairs, width = line_coordinates(X_aerial, X_street, G)
    for i in range(len(connected_pairs)):
        folium.PolyLine(connected_pairs[i], color="black", weight=width[i]*100, opacity=1).add_to(m)

    if satellite:
        folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Esri Satellite',
                overlay = False,
                control = True
            ).add_to(m)
    return(m)


def plot_predicted_mapping(COORDINATES, final, satellite = True):
    # initialize empty map zoomed in on 
    m = folium.Map(location=COORDINATES, zoom_start=13,tiles='CartoDBPositron')
    
    for i in range(final.shape[0]):
        #folium.Marker(each, icon=folium.Icon(color='blue', icon='cloud')).add_to(m)
        my_string = '<b>Name</b>: {}, <br><b>Year</b>: {}, <br><b>Tree_id</b>: {}, <br><b>Plot_id</b>: {},  <br><b>Diameter</b>: {},  <br><b>Height</b>: {}'.format(final.name[i], final.year[i], final.tree_id[i], final.plot_id[i], final.diameter[i], final.height[i])  
        folium.CircleMarker(location=[final.lat_x[i],final.lon_x[i]], radius=5, weight = 0, fill_color='blue', fill_opacity=0.8, popup = Popup(my_string)).add_to(m)

    if satellite:
        folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Esri Satellite',
                overlay = False,
                control = True
            ).add_to(m)
    return(m)