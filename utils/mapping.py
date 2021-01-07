import sklearn 
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import ot
import numpy as np
import scipy as sp

def nearest_neighbors(values, nbr_neighbors=15):
    nn = NearestNeighbors(nbr_neighbors, metric='euclidean', algorithm='brute').fit(values)
    dists, idxs = nn.kneighbors(values)
    return(dists, idxs)

def find_pair(X_ground_nn, X_drone_nn, annotations_files, ground_files):
    
    n = len(X_ground_nn)
    m = len(X_drone_nn)
    X = np.concatenate([X_ground_nn, X_drone_nn])
    
    dists, idxs = nearest_neighbors(X, nbr_neighbors=n+m)

    ground_index = []
    for i in range(n):
        p = 1
        id = X[i,2]
        while(X[idxs[i][p], 2] == id):
            p += 1
        j = idxs[i,p]
        ground_index.append(j-n)
    ground_index = np.array(ground_index, dtype = np.float32)
    
    ground_data = ground_files[['lat', 'lon', 'X', 'Y', 'name','year', 'tree_id', 'plot_id', 'diameter', 'height', 'is_musacea']]
    ground_data['ground_index'] = ground_index
    annotations_files['ground_index']=annotations_files.index
    final = pd.merge(annotations_files, ground_data, on='ground_index', how = 'inner', suffixes=('_d', '_g'))
    return(final)

def find_pair_2(X_ground_nn, X_drone_nn, annotations_files, ground_files):
    
    n = len(X_ground_nn)
    m = len(X_drone_nn)
    X = np.concatenate([X_drone_nn, X_ground_nn])
    
    dists, idxs = nearest_neighbors(X, nbr_neighbors=n+m)

    ground_index = []
    for i in range(m):
        p = 1
        id = X[i,2]
        while(X[idxs[i][p], 2] == id):
            p += 1
        j = idxs[i,p]
        ground_index.append(j-m)
    ground_index = np.array(ground_index, dtype = np.float32)
    
    ground_data = ground_files[['lat', 'lon', 'X', 'Y', 'name','year', 'tree_id', 'plot_id', 'diameter', 'height', 'is_musacea']]
    annotations_files['ground_index'] = ground_index
    ground_data['ground_index']=ground_data.index
    final = pd.merge(annotations_files, ground_data, on='ground_index', how = 'inner', suffixes=('_d', '_g'))
    return(final)


def binarycrossentropy(P_d, P_g):
    """
    P_d : probability of a drone tree to be a musacea; computed by the CNN
    P_g : probability of a ground tree to be a musacea; evaluated by humans; 0 or 1
    """
    n = len(P_d)
    m = len(P_g)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if P_g[j] == 1:
                if P_d[i] < 0.0001:
                    P_d[i] = 0.0001
                M[i][j] = - np.log(P_d[i])
            if P_g[j] == 0:
                if P_d[i] > 0.9999:
                    P_d[i] = 0.9999
                M[i][j] = - np.log(1-P_d[i])
    return(M)



def OT_emd(X_d, X_g):
  # loss matrix
  C = ot.dist(X_d, X_g)
  M = C/C.max()

  n = len(X_d)
  m = len(X_g)
  print(n, m)

  a, b = np.ones((n,)) / n, np.ones((m,)) / m

  G0 = ot.emd(a, b, M)

  plt.figure(3)
  plt.imshow(G0, interpolation='nearest')
  plt.title('OT matrix G0')

  plt.figure(4)
  ot.plot.plot2D_samples_mat(X_d[:,:2], X_g[:,:2], G0, c=[.5, .5, 1])
  plt.plot(X_d[:, 0], X_g[:, 1], '+b', label='Source samples')
  plt.plot(X_d[:, 0], X_g[:, 1], 'xr', label='Target samples')
  plt.legend(loc=0)
  plt.title('OT matrix with samples')


  return(G0)

def OT_scores_emd(X_d, X_g, P_d, P_g, mu = 0.5):
  # loss matrix
  # loss matrix
    C_dist = ot.dist(X_d, X_g)
    C_mus = binarycrossentropy(P_d, P_g)
    #ot.dist(np.expand_dims(scoresl, axis=1), np.expand_dims(scoresr, axis=1))
    #M_dist = C_dist/C_dist.max()
    #M_mus = C_mus/C_mus.max()

    M_dist = (C_dist-C_dist.min())/(C_dist.max()-C_dist.min())
    M_mus = (C_mus-C_mus.min())/(C_mus.max()-C_mus.min())
    
    M = M_dist + mu*M_mus

    n = len(X_d)
    m = len(X_g)
    print(n, m)

    a, b = np.ones((n,)) / n, np.ones((m,)) / m

    G0 = ot.emd(a, b, M)
    return(G0)



def OT_sinkhorn(X_d, X_g, lambd = 1e-2):
  # loss matrix
  C = ot.dist(X_d, X_g)
  M = C/C.max()

  n = len(X_d)
  m = len(X_g)
  print(n, m)

  a, b = np.ones((n,)) / n, np.ones((m,)) / m

  Gs = ot.bregman.sinkhorn_stabilized(a, b, M, lambd)

  """
  plt.figure(5)
  plt.imshow(Gs, interpolation='nearest')
  plt.title('OT matrix sinkhorn')

  plt.figure(6)
  ot.plot.plot2D_samples_mat(Xl[:,:2], Xr[:,:2], Gs, color=[.5, .5, 1])
  plt.plot(Xl[:, 0], Xl[:, 1], '+b', label='Source samples')
  plt.plot(Xr[:, 0], Xr[:, 1], 'xr', label='Target samples')
  plt.legend(loc=0)
  plt.title('OT matrix Sinkhorn with samples')
  """

  return(Gs)



def OT_scores_sinkhorn(X_d, X_g, P_d, P_g, mu = 0.5, lambd = 1e-2):
    # loss matrix
    C_dist = ot.dist(X_d, X_g)
    C_mus = binarycrossentropy(P_d, P_g)
    #ot.dist(np.expand_dims(scoresl, axis=1), np.expand_dims(scoresr, axis=1))
    #M_dist = C_dist/C_dist.max()
    #M_mus = C_mus/C_mus.max()

    M_dist = (C_dist-C_dist.min())/(C_dist.max()-C_dist.min())
    M_mus = (C_mus-C_mus.min())/(C_mus.max()-C_mus.min())
    
    M = M_dist + mu*M_mus

    n = len(X_d)
    m = len(X_g)
    print(n, m)

    a, b = np.ones((n,)) / n, np.ones((m,)) / m

    #Gs = ot.sinkhorn2(a, b, M, reg = lambd, method='sinkhorn_stabilized')
    Gs = ot.bregman.sinkhorn_stabilized(a, b, M, lambd)

    return(Gs)



def gromov_wasserstein(X_drone, X_ground):

    C1 = sp.spatial.distance.cdist(X_drone, X_drone)
    C2 = sp.spatial.distance.cdist(X_ground, X_ground)

    C1 /= C1.max()
    C2 /= C2.max()
    
    p = ot.unif(len(X_drone))
    q = ot.unif(len(X_ground))
    
    gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=True, log=True)
    
    return(gw0)

def gromov_wasserstein_entropic(X_drone, X_ground):

    C1 = sp.spatial.distance.cdist(X_drone, X_drone)
    C2 = sp.spatial.distance.cdist(X_ground, X_ground)

    C1 /= C1.max()
    C2 /= C2.max()
    
    p = ot.unif(len(X_drone))
    q = ot.unif(len(X_ground))
    
    gw, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=5e-2, log=True, verbose=True)
    
    return(gw)



def merge_drone_to_ground(annotations_files, ground_files, G):
    idx = np.argmax(G, axis = 1)
    annotations_files['ground_index']=idx
    ground_data = ground_files[['lat', 'lon', 'X', 'Y', 'name','year', 'tree_id', 'plot_id', 'diameter', 'height', 'is_musacea']]
    ground_data['ground_index'] = ground_data.index
    final = pd.merge(annotations_files, ground_data, on='ground_index', suffixes=('_d', '_g'))
    return(final)

def merge_ground_to_drone(annotations_files, ground_files, G):
    idx = np.argmax(G, axis = 0)
    ground_data = ground_files[['lat', 'lon', 'X', 'Y', 'name','year', 'tree_id', 'plot_id', 'diameter', 'height', 'is_musacea']]
    ground_data['ground_index'] = idx
    annotations_files['ground_index']=annotations_files.index
    final = pd.merge(annotations_files, ground_data, on='ground_index', how = 'inner', suffixes=('_d', '_g'))
    return(final)


def merge_greedy(annotations_files, ground_files, G):
    cost = -G*10e4
    row_ind, col_ind = linear_sum_assignment(cost)
    df_left = annotations_files.loc[row_ind]
    df_left['ground_index']= col_ind
    ground_data = ground_files[['lat', 'lon', 'X', 'Y', 'name','year', 'tree_id', 'plot_id', 'diameter', 'height', 'is_musacea']]
    ground_data['ground_index'] = ground_data.index
    final = pd.merge(df_left, ground_data, on='ground_index', how = 'inner', suffixes=('_d', '_g')) 
    return(final)

def acc_musacea(final):
    a = final.is_musacea_d.to_numpy()
    b = final.is_musacea_g.to_numpy()
    l = np.array([bool(elt) for elt in a])
    r = np.array([bool(elt) for elt in b])
    res = (r==l)
    acc = len(res[res==True])/len(res)
    return(acc)


def get_matching_baseline(list_sites):
    final = pd.DataFrame()
    
    for site in list_sites:
        
        path_file = 'annotations/{}_annotations_processed_cnn.csv'.format(site)
        annotations_files = pd.read_csv(path_file)
        annotations_files = annotations_files.reset_index(drop = True)
        
        ground_files = pd.read_csv('features/ground_data_{}.csv'.format(site), index_col = None)
        
        #X_ground = ground_files[["lat", "lon"]].to_numpy()
        #X_drone = annotations_files[["lat", "lon"]].to_numpy()
        
        X_ground = ground_files[["X", "Y"]].to_numpy()
        X_drone = annotations_files[["X", "Y"]].to_numpy()


        mus_ground = ground_files.is_musacea.to_numpy()
        mus_drone = annotations_files.is_musacea.to_numpy()
        
        X_drone_nn = np.hstack((X_drone, np.zeros((X_drone.shape[0], 1), dtype=np.int32)))
        X_ground_nn = np.hstack((X_ground, np.zeros((X_ground.shape[0], 1), dtype=np.int32)+1))
        final_site = find_pair(X_ground_nn, X_drone_nn, annotations_files, ground_files)
        
        final = pd.concat([final, final_site])
    
    final = final.reset_index(drop = True)
    return(final)



def get_map(list_sites, method, ot = 'sinkhorn', lambd = 0.01, mu = 1):
    G_big = []
    
    for site in list_sites:
        
        path_file = 'annotations/{}_annotations_processed_cnn.csv'.format(site)
        annotations_files = pd.read_csv(path_file)
        annotations_files = annotations_files.reset_index(drop = True)
        
        ground_files = pd.read_csv('features/ground_data_{}.csv'.format(site), index_col = None)
        
        #X_ground = ground_files[["lat", "lon"]].to_numpy()
        #X_drone = annotations_files[["lat", "lon"]].to_numpy()
        
        X_ground = ground_files[["X", "Y"]].to_numpy()
        X_drone = annotations_files[["X", "Y"]].to_numpy()

        mus_ground = ground_files.is_musacea.to_numpy()
        mus_drone = annotations_files.is_musacea.to_numpy()
        
        if method == 'OT':
            if ot == 'emd':
                G = OT_emd(X_drone, X_ground)
            if ot == 'sinkhorn':
                G = OT_sinkhorn(X_drone, X_ground, lambd = lambd)
            
        if method == 'OT + CNN':
            if ot == 'emd':
                G = OT_scores_emd(X_drone, X_ground, mus_drone, mus_ground, mu = mu)
            if ot == 'sinkhorn':
                G = OT_scores_sinkhorn(X_drone, X_ground, mus_drone, mus_ground, mu = mu, lambd = lambd)
             
        if method == 'GW':
            G = gromov_wasserstein(X_drone, X_ground)

        if method == 'Entropic GW':
            G = gromov_wasserstein_entropic(X_drone, X_ground)
            
        G_big.append(G)
    return(G_big)


def get_matching(list_sites, G_big, greedy, drone_to_ground = False):
    final = pd.DataFrame()
    
    for i in range(len(list_sites)):
        site = list_sites[i]

        path_file = 'annotations/{}_annotations_processed_cnn.csv'.format(site)
        annotations_files = pd.read_csv(path_file)
        annotations_files = annotations_files.reset_index(drop = True)
        
        ground_files = pd.read_csv('features/ground_data_{}.csv'.format(site), index_col = None)
        
        if greedy == False:
            if drone_to_ground == False:
                final_site = merge_ground_to_drone(annotations_files, ground_files, G_big[i])
            if drone_to_ground == True:
                final_site = merge_drone_to_ground(annotations_files, ground_files, G_big[i])
        
        if greedy == True:
            final_site = merge_greedy(annotations_files, ground_files, G_big[i])
        
        final = pd.concat([final, final_site])
    
    final = final.reset_index(drop = True)
    return(final)