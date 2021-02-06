import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import scipy as sp
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(values, nbr_neighbors=15):
    nn = NearestNeighbors(nbr_neighbors, metric='euclidean', algorithm='brute').fit(values)
    dists, ids = nn.kneighbors(values)
    return dists, ids


def find_pair(x_ground_nn, x_drone_nn, annotations_files, ground_files):
    """
    Connect each ground data to its 1-nearest neighbour among the drone data points.
    Args:
        x_ground_nn (numpy array): ground data position (latitude, longitude), identified by id = 0
        x_drone_nn (numpy array): drone data position (latitude, longitude), identified by id = 1
        annotations_files (dataframe): drone data
        ground_files (dataframe): ground data

    Returns:
        final (dataframe): merge ground and drone data following the nearest neighbours mapping
    """
    n = len(x_ground_nn)
    m = len(x_drone_nn)
    x = np.concatenate([x_ground_nn, x_drone_nn])

    dists, ids = nearest_neighbors(x, nbr_neighbors=n + m)

    ground_index = []
    for i in range(n):
        p = 1
        id = x[i, 2]
        while x[ids[i][p], 2] == id:
            p += 1
        j = ids[i, p]
        ground_index.append(j - n)
    ground_index = np.array(ground_index, dtype=np.float32)

    ground_data = ground_files
    ground_data['ground_index'] = ground_index
    ground_data['id'] = ground_data.index
    annotations_files['ground_index'] = annotations_files.index
    final = pd.merge(annotations_files, ground_data, on='ground_index', how='inner', suffixes=('_d', '_g'))
    return final


def find_pair_2(x_ground_nn, x_drone_nn, annotations_files, ground_files):
    """
    Connect each drone data to its 1-nearest neighbour among the ground data points.
    Args:
        x_ground_nn (numpy array): ground data position (latitude, longitude), identified by id = 0
        x_drone_nn (numpy array): drone data position (latitude, longitude), identified by id = 1
        annotations_files (dataframe): drone data
        ground_files (dataframe): ground data

    Returns:
        final (dataframe): merge ground and drone data following the nearest neighbours mapping
    """
    n = len(x_ground_nn)
    m = len(x_drone_nn)
    x = np.concatenate([x_drone_nn, x_ground_nn])

    dists, ids = nearest_neighbors(x, nbr_neighbors=n + m)

    ground_index = []
    for i in range(m):
        p = 1
        id = x[i, 2]
        while x[ids[i][p], 2] == id:
            p += 1
        j = ids[i, p]
        ground_index.append(j - m)
    ground_index = np.array(ground_index, dtype=np.float32)

    ground_data = ground_files[
        ['lat', 'lon', 'X', 'Y', 'name', 'year', 'tree_id', 'plot_id', 'diameter', 'height', 'is_musacea']]
    annotations_files['ground_index'] = ground_index
    ground_data['ground_index'] = ground_data.index
    final = pd.merge(annotations_files, ground_data, on='ground_index', how='inner', suffixes=('_d', '_g'))
    return final


def binary_crossentropy(p_d, p_g):
    """
    Computes the binary cross-entropy between two probability distributions
    Args:
        p_d (list): list of probabilities of a drone tree to be a musacea; computed by the CNN
        p_g (list): list of probabilities of a ground tree to be a musacea; evaluated by humans; 0 or 1

    Returns:
        M: matrix of binary cross-entropies
    """
    n = len(p_d)
    m = len(p_g)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if p_g[j] == 1:
                if p_d[i] < 0.0001:
                    p_d[i] = 0.0001
                M[i][j] = - np.log(p_d[i])
            if p_g[j] == 0:
                if p_d[i] > 0.9999:
                    p_d[i] = 0.9999
                M[i][j] = - np.log(1 - p_d[i])
    return M


def OT_emd(X_d, X_g):
    """
    Computes the optimal transport plan using the Earth Mover's distance (discrete Wasserstein distance)
    Args:
        X_d (numpy array): drone positions (lat, lon)
        X_g (numpy array): ground positions (lat, lon)

    Returns:
        g0 (numpy array): optimal transport plan
    """
    # loss matrix
    C = ot.dist(X_d, X_g)
    M = C / C.max()

    n = len(X_d)
    m = len(X_g)
    print(n, m)

    a, b = np.ones((n,)) / n, np.ones((m,)) / m

    g0 = ot.emd(a, b, M)

    plt.figure(3)
    plt.imshow(g0, interpolation='nearest')
    plt.title('OT matrix G0')

    plt.figure(4)
    ot.plot.plot2D_samples_mat(X_d[:, :2], X_g[:, :2], g0, c=[.5, .5, 1])
    plt.plot(X_d[:, 0], X_g[:, 1], '+b', label='Source samples')
    plt.plot(X_d[:, 0], X_g[:, 1], 'xr', label='Target samples')
    plt.legend(loc=0)
    plt.title('OT matrix with samples')

    return (g0)


def OT_scores_emd(X_d, X_g, p_d, p_g, mu=0.5):
    """
    Computes the optimal transport plan using the Earth Mover's distance (discrete Wasserstein distance) on GPS position
    and initial tree species prediction (p_d, p_g)
    Args:
        X_d (numpy array): drone positions (lat, lon)
        X_g (numpy array): ground positions (lat, lon)
        p_d (list): list of probabilities of the tree from drone data
        p_g (list): list of probabilities of the tree from ground data
        mu (float): balancing coefficient between terms of the cost

    Returns:
        g0 (numpy array): optimal transport plan
    """
    # loss matrix
    # loss matrix
    C_dist = ot.dist(X_d, X_g)
    C_mus = binary_crossentropy(p_d, p_g)
    # ot.dist(np.expand_dims(scoresl, axis=1), np.expand_dims(scoresr, axis=1))
    # M_dist = C_dist/C_dist.max()
    # M_mus = C_mus/C_mus.max()

    M_dist = (C_dist - C_dist.min()) / (C_dist.max() - C_dist.min())
    M_mus = (C_mus - C_mus.min()) / (C_mus.max() - C_mus.min())

    M = M_dist + mu * M_mus

    n = len(X_d)
    m = len(X_g)
    print(n, m)

    a, b = np.ones((n,)) / n, np.ones((m,)) / m

    g0 = ot.emd(a, b, M)
    return (g0)


def OT_sinkhorn(X_d, X_g, lambd=1e-2):
    """
    Computes the optimal transport plan using the Sinkhorn distance on GPS position
    Args:
        X_d (numpy array): drone positions (lat, lon)
        X_g (numpy array): ground positions (lat, lon)
        lambd (float): coefficient for the entropy term

    Returns:
        gs: optimal transport plan
    """
    # loss matrix
    C = ot.dist(X_d, X_g)
    M = C / C.max()

    n = len(X_d)
    m = len(X_g)
    print(n, m)

    a, b = np.ones((n,)) / n, np.ones((m,)) / m

    gs = ot.bregman.sinkhorn_stabilized(a, b, M, lambd)
    return (gs)


def OT_scores_sinkhorn(X_d, X_g, p_d, p_g, mu=0.5, lambd=1e-2):
    """
    Computes the optimal transport plan using the Sinkhorn distance on GPS position
    and initial tree species prediction (p_d, p_g)
    Args:
        X_d: drone positions (lat, lon)
        X_g: ground positions (lat, lon)
        p_d: list of probabilities of the tree from drone data
        p_g: list of probabilities of the tree from ground data
        mu: balancing coefficient between terms of the cost
        lambd: coefficient for the entropy term

    Returns:
        gs: optimal transport plan
    """
    # loss matrix
    C_dist = ot.dist(X_d, X_g)
    C_mus = binary_crossentropy(p_d, p_g)
    # ot.dist(np.expand_dims(scoresl, axis=1), np.expand_dims(scoresr, axis=1))
    # M_dist = C_dist/C_dist.max()
    # M_mus = C_mus/C_mus.max()

    M_dist = (C_dist - C_dist.min()) / (C_dist.max() - C_dist.min())
    M_mus = (C_mus - C_mus.min()) / (C_mus.max() - C_mus.min())

    M = M_dist + mu * M_mus

    n = len(X_d)
    m = len(X_g)
    print(n, m)

    a, b = np.ones((n,)) / n, np.ones((m,)) / m

    # Gs = ot.sinkhorn2(a, b, M, reg = lambd, method='sinkhorn_stabilized')
    gs = ot.bregman.sinkhorn_stabilized(a, b, M, lambd)

    return (gs)


def gromov_wasserstein(X_d, X_g):
    """
    Computes the optimal transport plan using the Gromov-Wasserstein distance on GPS position
    Args:
        X_d (numpy array): drone positions (lat, lon)
        X_g (numpy array): ground positions (lat, lon)

    Returns:
        gw0 (numpy array): optimal transport plan
    """
    C1 = sp.spatial.distance.cdist(X_d, X_d)
    C2 = sp.spatial.distance.cdist(X_g, X_g)

    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(len(X_d))
    q = ot.unif(len(X_g))

    gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=True, log=True)

    return (gw0)


def gromov_wasserstein_entropic(X_d, X_g):
    """
    Computes the optimal transport plan using the Gromov-Wasserstein distance on GPS position with an entropy term
    Args:
        X_d (numpy array): drone positions (lat, lon)
        X_g (numpy array): ground positions (lat, lon)

    Returns:
        gw0 (numpy array): optimal transport plan
    """
    C1 = sp.spatial.distance.cdist(X_d, X_d)
    C2 = sp.spatial.distance.cdist(X_g, X_g)

    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(len(X_d))
    q = ot.unif(len(X_g))

    gw, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=5e-2, log=True, verbose=True)

    return (gw)


def merge_drone_to_ground(annotations_files, ground_files, G):
    """
    Merge all drone data points to ground data
    Args:
        annotations_files (dataframe): drone data
        ground_files (dataframe): ground data
        G (numpy array): optimal transport plan

    Returns:
        final (dataframe): final dataframe - tree dataset
    """
    idx = np.argmax(G, axis=1)
    annotations_files['ground_index'] = idx
    ground_data = ground_files
    ground_data['ground_index'] = ground_data.index
    ground_data['id'] = ground_data.index
    final = pd.merge(annotations_files, ground_data, on='ground_index', suffixes=('_d', '_g'))
    return (final)


def merge_ground_to_drone(annotations_files, ground_files, G):
    """
    Merge all ground data points to drone data (all ground measurements are assigned to trees detected on drone images)
    Args:
        annotations_files: drone data
        ground_files: ground data
        G: optimal transport plan

    Returns:
        final: final dataframe - tree dataset
    """
    idx = np.argmax(G, axis=0)
    ground_data = ground_files
    ground_data['ground_index'] = idx
    ground_data['id'] = ground_data.index
    annotations_files['ground_index'] = annotations_files.index
    final = pd.merge(annotations_files, ground_data, on='ground_index', how='inner', suffixes=('_d', '_g'))
    return (final)


def merge_greedy(annotations_files, ground_files, G):
    """
    Merge ground and drone data in a greedy manner: the data points of the smallest dataset are all allocated to
    data points of the largest dataset; each assignment is unique: no possible double matching of a data point.
    Args:
        annotations_files: drone data
        ground_files: ground data
        G: optimal transport plan

    Returns:
        final: final dataframe - tree dataset
    """
    cost = -G * 10e4
    row_ind, col_ind = linear_sum_assignment(cost)
    df_left = annotations_files.loc[row_ind]
    df_left['ground_index'] = col_ind
    ground_data = ground_files
    # [['lat', 'lon', 'X', 'Y', 'name', 'year', 'tree_id', 'plot_id', 'diameter', 'height', 'is_musacea']]
    ground_data['ground_index'] = ground_data.index
    ground_data['id'] = ground_data.index
    final = pd.merge(df_left, ground_data, on='ground_index', how='inner', suffixes=('_d', '_g'))
    return final


def acc_musacea(final):
    """
    Computes the accuracy of the final tree dataset after mapping ground and drone data
    Args:
        final (dataframe): tree dataset obtained after mapping

    Returns:
        acc (float): accuracy of the mapping
    """
    a = final.is_musacea_d.to_numpy()
    b = final.is_musacea_g.to_numpy()
    L = np.array([bool(elt) for elt in a])
    r = np.array([bool(elt) for elt in b])
    res = (r == L)
    acc = len(res[res == True]) / len(res)
    return acc


"""
====== Ecuador ======
Three functions to map ground and drone data, and create the final tree datasets for Ecuador dataset.
"""


def get_matching_baseline_Ecuador(list_sites):
    """
    Returns the mapping (tree dataset) following the nearest neighbours strategy
    Args:
        list_sites: list of sites in Ecuador

    Returns:
        final: final tree dataset for all sites combined
    """
    final = pd.DataFrame()

    for site in list_sites:
        path_file = 'Ecuador/annotations/{}_annotations_processed_cnn.csv'.format(site)
        annotations_files = pd.read_csv(path_file)
        annotations_files = annotations_files.reset_index(drop=True)

        ground_files = pd.read_csv('Ecuador/features/ground_data_{}.csv'.format(site), index_col=None)

        # X_ground = ground_files[["lat", "lon"]].to_numpy()
        # X_drone = annotations_files[["lat", "lon"]].to_numpy()

        X_ground = ground_files[["X", "Y"]].to_numpy()
        X_drone = annotations_files[["X", "Y"]].to_numpy()

        X_drone_nn = np.hstack((X_drone, np.zeros((X_drone.shape[0], 1), dtype=np.int32)))
        X_ground_nn = np.hstack((X_ground, np.zeros((X_ground.shape[0], 1), dtype=np.int32) + 1))
        final_site = find_pair(X_ground_nn, X_drone_nn, annotations_files, ground_files)

        final = pd.concat([final, final_site])

    final = final.reset_index(drop=True)
    return final


def get_map_Ecuador(list_sites, method, ot_type='sinkhorn', lambd=0.01, mu=1):
    """
    Compute the optimal transport plan for each site independently
    Args:
        list_sites: list of sites in the dataset
        method: mapping method
        ot_type: if method is optimal transport, type of cost/distance
        lambd: coefficient for the entropy term
        mu: balancing coefficient between terms of the cost

    Returns:
        G_big: list of optimal transport plans for each site
    """
    G_big = []

    for site in list_sites:

        path_file = 'annotations/{}_annotations_processed_cnn.csv'.format(site)
        annotations_files = pd.read_csv(path_file)
        annotations_files = annotations_files.reset_index(drop=True)

        ground_files = pd.read_csv('Ecuador/features/ground_data_{}.csv'.format(site), index_col=None)

        # X_ground = ground_files[["lat", "lon"]].to_numpy()
        # X_drone = annotations_files[["lat", "lon"]].to_numpy()

        X_ground = ground_files[["X", "Y"]].to_numpy()
        X_drone = annotations_files[["X", "Y"]].to_numpy()

        mus_ground = ground_files.is_musacea.to_numpy()
        mus_drone = annotations_files.is_musacea.to_numpy()

        if method == 'OT':
            if ot_type == 'emd':
                G = OT_emd(X_drone, X_ground)
            if ot_type == 'sinkhorn':
                G = OT_sinkhorn(X_drone, X_ground, lambd=lambd)

        if method == 'OT + CNN':
            if ot_type == 'emd':
                G = OT_scores_emd(X_drone, X_ground, mus_drone, mus_ground, mu=mu)
            if ot_type == 'sinkhorn':
                G = OT_scores_sinkhorn(X_drone, X_ground, mus_drone, mus_ground, mu=mu, lambd=lambd)

        if method == 'GW':
            G = gromov_wasserstein(X_drone, X_ground)

        if method == 'Entropic GW':
            G = gromov_wasserstein_entropic(X_drone, X_ground)

        G_big.append(G)
    return G_big


def get_matching_Ecuador(list_sites, G_big, greedy, drone_to_ground=False):
    """
    Deduce from the optimal transport plans in G_big the final tree datasets according to the merging strategy
    Args:
        list_sites: list of the sites in the datasets
        G_big: list of mapping matrices for each site
        greedy: merging strategy
        drone_to_ground: if True, connect all drone data points; else, connect all ground data points

    Returns:
        final: final tree dataset for all sites combined
    """
    final = pd.DataFrame()

    for i in range(len(list_sites)):
        site = list_sites[i]

        path_file = 'Ecuador/annotations/{}_annotations_processed_cnn.csv'.format(site)
        annotations_files = pd.read_csv(path_file)
        annotations_files = annotations_files.reset_index(drop=True)

        ground_files = pd.read_csv('Ecuador/features/ground_data_{}.csv'.format(site), index_col=None)

        if greedy is False:
            if drone_to_ground is False:
                final_site = merge_ground_to_drone(annotations_files, ground_files, G_big[i])
            else:
                final_site = merge_drone_to_ground(annotations_files, ground_files, G_big[i])

        else:
            final_site = merge_greedy(annotations_files, ground_files, G_big[i])

        final = pd.concat([final, final_site])

    final = final.reset_index(drop=True)
    return final


"""
====== NEON ======
Three functions to map ground and drone data, and create the final tree datasets for NEON dataset.
"""


def get_matching_baseline_NEON(list_sites, df_drone, df_ground):
    """
    Returns the mapping following the nearest neighbours strategy.
    Args:
        list_sites (list): list of the sites in the datasets
        df_drone (dataframe): drone data
        df_ground (dataframe): ground data

    Returns:
        final (dataframe): final tree dataset for all sites combined
    """
    final = pd.DataFrame()

    for site in list_sites:
        annotations_files = df_drone[df_drone.site == site]
        annotations_files = annotations_files.reset_index(drop=True)

        ground_files = df_ground[df_ground.site == site]
        ground_files = ground_files.reset_index(drop=True)

        X_ground = ground_files[["X", "Y"]].to_numpy()
        X_drone = annotations_files[["X", "Y"]].to_numpy()

        X_drone_nn = np.hstack((X_drone, np.zeros((X_drone.shape[0], 1), dtype=np.int32)))
        X_ground_nn = np.hstack((X_ground, np.zeros((X_ground.shape[0], 1), dtype=np.int32) + 1))
        final_site = find_pair(X_ground_nn, X_drone_nn, annotations_files, ground_files)

        final = pd.concat([final, final_site])

    final = final.reset_index(drop=True)
    return final


def get_map_NEON(list_sites, df_drone, df_ground, method, ot='sinkhorn', lambd=0.01, mu=1):
    G_big = []

    for i in range(len(list_sites)):
        site = list_sites[i]
        annotations_files = df_drone[df_drone.site == site]
        annotations_files = annotations_files.reset_index(drop=True)

        ground_files = df_ground[df_ground.site == site]
        ground_files = ground_files.reset_index(drop=True)

        X_ground = ground_files[["X", "Y"]].to_numpy()
        X_drone = annotations_files[["X", "Y"]].to_numpy()

        if method == 'OT':
            if ot == 'emd':
                G = OT_emd(X_drone, X_ground)
            if ot == 'sinkhorn':
                G = OT_sinkhorn(X_drone, X_ground, lambd=lambd)

        if method == 'GW':
            G = gromov_wasserstein(X_drone, X_ground)

        if method == 'Entropic GW':
            G = gromov_wasserstein_entropic(X_drone, X_ground)

        G_big.append(G)
    return G_big


def get_matching_NEON(list_sites, df_drone, df_ground, G_big, greedy, drone_to_ground=False):
    """
    Deduces from the optimal transport plans in G_big the final tree datasets according to the merging strategy
    Args:
        list_sites (list): list of the sites in the datasets
        df_drone (dataframe): drone data
        df_ground (dataframe): ground data
        G_big (numpy array): list of mapping matrices for each site
        greedy (bool): merging strategy. If True, the merging is greedy.
        drone_to_ground (bool): if True, connect all drone data points; else, connect all ground data points

    Returns:
        final (dataframe): final tree dataset for all sites combined
    """

    final = pd.DataFrame()

    for i in range(len(list_sites)):
        site = list_sites[i]
        annotations_files = df_drone[df_drone.site == site]
        annotations_files = annotations_files.reset_index(drop=True)

        ground_files = df_ground[df_ground.site == site]
        ground_files = ground_files.reset_index(drop=True)

        if greedy is False:
            if drone_to_ground is False:
                final_site = merge_ground_to_drone(annotations_files, ground_files, G_big[i])
            else:
                final_site = merge_drone_to_ground(annotations_files, ground_files, G_big[i])

        else:
            final_site = merge_greedy(annotations_files, ground_files, G_big[i])

        final = pd.concat([final, final_site])

    final = final.reset_index(drop=True)
    return final
