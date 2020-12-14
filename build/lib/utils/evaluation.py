import numpy as np


# Compute final predicted mapping 

def mapping_pred(G):
  M = np.zeros((G.shape[0], G.shape[1]))
  row_ind = np.argmax(G, axis = 0)
  M[row_ind, :] = 1
  M = M.astype(int)
  return(M)


# Compute true mapping
def mapping_true(df1, df2):
  M = np.zeros((df1.shape[0], df2.shape[0]))
  for index, row in df1.iterrows():
   col_ind = df2[df2.item_id == row.target].index[0]
   M[index, col_ind] = 1
  return(M)


 # Compute accuracy of the matching
def accuracy(M_pred, M_true):
  acc = np.sum(np.multiply(M_true, M_pred))/np.sum(M_true)
  return(acc)