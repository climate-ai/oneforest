
import pandas as pd
import geopandas as gpd


def extract_ground_data(ground_dir, annot_dir):
    # Extract citizen science
    # For each tree (reported), we have GPS coordinates, species, size,..

    # We keep as features for each Tree: 
    # Name: 'Variedad_1', 
    # Lat:'_Gps_1_lat', 
    # Lon:'_Gps_1_lon', 
    # Average Diameter: 'Ave_Diamet', 
    # Height of Tree: 'Altura_del', 
    # Year when the tree was planted: 'Plant_Yr'
    # Plot identifier: Plot
    # Index of Tree: _index

    shapefile = gpd.read_file(ground_dir)
    ground_files = pd.DataFrame(shapefile)
    print(ground_files.columns)

    column_names = ["name",  "lat", "lon", "diameter", 'height', 'year', 'plot_id', 'tree_id']
    ground_files.rename(columns={'Variedad_1' : "name",  '_Gps_1_lat' : "lat", '_Gps_1_lon' : "lon",  
                    'Ave_Diamet': "diameter", 'Altura_del':'height', 'Plant_Yr':'year', 'Plot':'plot_id', '_index': 'tree_id'},inplace=True)
    ground_files = ground_files[column_names]
    ground_files.to_csv('{}/ground_data.csv'.format(annot_dir), index = False)


if __name__ == '__main__':
    ground_dir = "wwf_ecuador/Final_Trees/Final_Trees.shp"
    annot_dir = 'annotations'

    extract_ground_data(ground_dir, annot_dir)
