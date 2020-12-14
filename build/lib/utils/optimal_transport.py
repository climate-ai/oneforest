import matplotlib.pyplot as plt
import ot
import numpy as np


def OT_emd(Xl, Xr):
  # loss matrix
  C = ot.dist(Xl, Xr)
  M = C/C.max()

  n = len(Xl)
  m = len(Xr)
  print(n, m)

  a, b = np.ones((n,)) / n, np.ones((m,)) / m

  G0 = ot.emd(a, b, M)

  plt.figure(3)
  plt.imshow(G0, interpolation='nearest')
  plt.title('OT matrix G0')

  plt.figure(4)
  ot.plot.plot2D_samples_mat(Xl[:,:2], Xr[:,:2], G0, c=[.5, .5, 1])
  plt.plot(Xl[:, 0], Xl[:, 1], '+b', label='Source samples')
  plt.plot(Xr[:, 0], Xr[:, 1], 'xr', label='Target samples')
  plt.legend(loc=0)
  plt.title('OT matrix with samples')

  return(G0)

def OT_scores_emd(Xl, Xr, scoresl, scoresr, mu = 0.5):
  # loss matrix
  print(scoresl)
  C = ot.dist(Xl, Xr) + mu* ot.dist(np.expand_dims(scoresl, axis=1), np.expand_dims(scoresr, axis=1))
  M = C/C.max()

  n = len(Xl)
  m = len(Xr)
  print(n, m)

  a, b = scoresl, scoresr

  G0 = ot.emd(a, b, M)

  plt.figure(3)
  plt.imshow(G0, interpolation='nearest')
  plt.title('OT matrix G0')

  plt.figure(4)
  ot.plot.plot2D_samples_mat(Xl[:,:2], Xr[:,:2], G0, c=[.5, .5, 1])
  plt.plot(Xl[:, 0], Xl[:, 1], '+b', label='Source samples')
  plt.plot(Xr[:, 0], Xr[:, 1], 'xr', label='Target samples')
  plt.legend(loc=0)
  plt.title('OT matrix with samples')

  return(G0)



def OT_sinkhorn(Xl, Xr, lambd = 1e-2):
  # loss matrix
  C = ot.dist(Xl, Xr) + ll * ot.dist(hl, hr)
  M = C/C.max()

  n = len(Xl)
  m = len(Xr)
  print(n, m)

  a, b = np.ones((n,)) / n, np.ones((m,)) / m

  Gs = ot.sinkhorn(a, b, M, lambd)

  plt.figure(5)
  plt.imshow(Gs, interpolation='nearest')
  plt.title('OT matrix sinkhorn')

  plt.figure(6)
  ot.plot.plot2D_samples_mat(Xl[:,:2], Xr[:,:2], Gs, color=[.5, .5, 1])
  plt.plot(Xl[:, 0], Xl[:, 1], '+b', label='Source samples')
  plt.plot(Xr[:, 0], Xr[:, 1], 'xr', label='Target samples')
  plt.legend(loc=0)
  plt.title('OT matrix Sinkhorn with samples')

  return(Gs)

def normalize(a):
  e = (a - np.mean(a)) / np.std(a)
  return(e)

def OT_scores_sinkhorn(Xl, Xr, scoresl, scoresr, mu = 0.5, lambd = 1e-2):
  # loss matrix
  C = ot.dist(Xl, Xr) + mu * ot.dist(np.expand_dims(scoresl, axis=1), np.expand_dims(scoresr, axis=1))
  M = C/C.max()

  n = len(Xl)
  m = len(Xr)
  print(n, m)

  a, b = scoresl, scoresr

  Gs = ot.sinkhorn(a, b, M, lambd)

  plt.figure(5)
  plt.imshow(Gs, interpolation='nearest')
  plt.title('OT matrix sinkhorn')

  plt.figure(6)
  ot.plot.plot2D_samples_mat(Xl[:,:2], Xr[:,:2], Gs, color=[.5, .5, 1])
  plt.plot(Xl[:, 0], Xl[:, 1], '+b', label='Source samples')
  plt.plot(Xr[:, 0], Xr[:, 1], 'xr', label='Target samples')
  plt.legend(loc=0)
  plt.title('OT matrix Sinkhorn with samples')

  return(Gs)