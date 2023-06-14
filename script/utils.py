import numpy as np
import random


def generate_cr_missing(mat,lost_rate,mode='c'):
  """
  Generate colume(row)-wise entire missing entries to simulate the unmeasured scenarios in kriging problem
  input:
    mat: spatiotemporal matrix with shape (time*location)
  """
    if mode == 'c':
        column_nums = mat.shape[1]
        lost_columns = random.sample(list(range(column_nums)),int(lost_rate*column_nums))
        matrix_cr_lost = mat.copy()
        matrix_cr_lost[:,lost_columns] = 0
    elif mode == 'r':
        row_nums = mat.shape[0]
        lost_rows = random.sample(list(range(row_nums)),int(lost_rate*row_nums))
        matrix_cr_lost = mat.copy()
        matrix_cr_lost[lost_rows,:] = 0
    else:
        raise TypeError(" 'mode' need to be 'c' or 'r'! ")
    
    return matrix_cr_lost


def generate_random_missing(link_matrix,lost_rate): 
   """Generate element-wise random missing"""
    link_matrix_lost = link_matrix.copy()
    coord = []
    m,n = link_matrix.shape
    for i in range(m):
        for j in range(n):
            coord.append((i,j))
    
    mask = random.sample(coord,int(lost_rate*len(coord)))
    for coord in mask:
        link_matrix_lost[coord[0],coord[1]] = 0
      
    return link_matrix_lost

def get_missing_rate(X_lost):
    o_channel_num = (X_lost == 0).astype(int).sum().sum()
    matrix_miss_rate = o_channel_num/(X_lost.size)
    
    return matrix_miss_rate


def compute_MAE(X_masked,X_true,X_hat):
   #Only calculate the errors on the masked and nonzero positions
    pos_test = np.where((X_true != 0) & (X_masked == 0))
    MAE = np.sum(abs(X_true[pos_test]-X_hat[pos_test]))/X_true[pos_test].shape[0]
    
    return MAE


def compute_RMSE(X_masked,X_true,X_hat):
    pos_test = np.where((X_true != 0) & (X_masked == 0))
    RMSE = np.sqrt(((X_true[pos_test]-X_hat[pos_test])**2).sum()/X_true[pos_test].shape[0])
    
    return RMSE


def compute_MAPE(X_masked,X_true,X_hat): 
    pos_test = np.where((X_true != 0) & (X_masked == 0))
    MAPE = np.sum(np.abs(X_true[pos_test]-X_hat[pos_test]) / X_true[pos_test]) / X_true[pos_test].shape[0]
    
    return MAPE
