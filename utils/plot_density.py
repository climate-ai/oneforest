import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_angle(a, b):
    c = np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))
    return(np.degrees(np.arccos(c)))

def get_distance(a, b):
    dst = np.linalg.norm(a-b)
    return(dst)


def density_plots(title, final, bw = .25):
    
    final['distance'] = final.apply(lambda x: get_distance(np.array([x.lat_d, x.lon_d]),np.array([x.lat_g, x.lon_g])), axis = 1)
    dist = final['distance'].to_numpy()*1e6
    
    final['angle'] = final.apply(lambda x: get_angle(np.array([x.lat_d, x.lon_d]),np.array([x.lat_g, x.lon_g])), axis = 1)
    angle = final['angle'].to_numpy()*1e6

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title,fontweight="bold", fontsize=20, y=1.05)

    # Distance
    sns.kdeplot(dist, ax = axes[0], bw_adjust=bw)
    axes[0].set_title('Transport Distance', fontsize = 16)
    axes[0].set_xlabel('Distance x(1e-6)')
    # Angle
    sns.kdeplot(angle, ax = axes[1], bw_adjust=bw)
    axes[1].set_title('Transport Angle', fontsize = 16)
    axes[1].set_xlabel('Angle (degrees)x(1e-6)')

    return(fig)