import numpy as np
import pandas as pd

import datetime
import folium
from folium.map import *
from folium import plugins
from folium.plugins import MeasureControl
from folium.plugins import FloatImage

def merge(annotations_files, ground_files, G):
    idx = np.argmax(G, axis = 1)
    annotations_files['ground_index']=idx
    ground_data = ground_files[['name','year', 'tree_id', 'plot_id', 'diameter', 'height']]
    ground_data['ground_index'] = ground_data.index
    final = pd.merge(annotations_files, ground_data, on='ground_index')
    return final

def plot_predicted_mapping(COORDINATES, final, satellite = True):
    # initialize empty map zoomed in on 
    m = folium.Map(location=COORDINATES, zoom_start=13,tiles='CartoDBPositron')
    
    for i in range(final.shape[0]):
        #folium.Marker(each, icon=folium.Icon(color='blue', icon='cloud')).add_to(m)
        my_string = '<b>Name</b>: {}, <br><b>Year</b>: {}, <br><b>Tree_id</b>: {}, <br><b>Plot_id</b>: {},  <br><b>Diameter</b>: {},  <br><b>Height</b>: {}'.format(df_street.name[i], df_street.year[i], df_street.tree_id[i], df_street.plot_id[i], df_street.diameter[i], df_street.height[i])  
        folium.CircleMarker(location=[final.lat[i],final.lon[i]], radius=5, weight = 0, fill_color='blue', fill_opacity=0.8, popup = Popup(my_string)).add_to(m)

    if satellite:
        folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Esri Satellite',
                overlay = False,
                control = True
            ).add_to(m)
    return(m)