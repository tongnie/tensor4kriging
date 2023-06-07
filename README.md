## Spatiotemporal Laplacian-enhanced low-rank tensor learning for large-scale traffic speed kriging with missing data
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)


## To be completed
> This is the code repository for our paper 'Correlating sparse sensing for large-scale traffic speed estimation: A Laplacian-enhanced low-rank tensor kriging approach' that will be published on Transportation Research Part C. The preprint version is available at [arXiv](https://arxiv.org/abs/2210.11780).


## Motivation
Traffic speed is central to characterizing the fluidity of the road network. Many transportation applications rely
on it, such as real-time navigation, dynamic route planning, and congestion management. However, due to sparse
deployment of static sensors or low penetration of mobile sensors, speeds detected are incomplete and far from
network-wide use. In addition, sensors are prone to error or missing data due to various kinds of reasons, speeds
from these sensors can become highly noisy. These drawbacks call for effective techniques to recover credible
estimates from the incomplete data. In this repository we demonstrated a **L**aplacian-**e**nhanced low-rank **t**ensor **c**ompletion (LETC) framework featuring both low-rankness and multi-dimensional correlations for large-scale traffic speed kriging under limited observations. 

<p align="center">
<img align="middle" src="graphics/Fig1.png" alt="fig1" width="700">
</p>


## Dataset
We adopt the large-scale PeMS-4W data to demonstrate how to implement LETC model to perform kriging with missing data imputation.
- **PeMS-4W**: Large-scale traffic speed data measured by 11160 static sensors from the [performance measurement system](https://pems.dot.ca.gov/) in California. The first four weeks of loop speed data with a 5-min window is pre-processed and available at [zenodo](https://zenodo.org/record/3939793).

Load graph information:

```python
def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

sid, sind, adj = load_graph_data('California-data-set/adj_mat.pkl')
```

Prepare tensors:

```python
import numpy as np
import pandas as pd

tensor = np.load('xxx.npz')

```

## Model implementation

<p align="center">
<img align="middle" src="graphics/Fig2.png" alt="fig2" width="700">
</p>


Our model is based on a consise NumPy implementation on CPU devices, which is also applicable with CuPy on a GPU device. Some key operations are discussed as below:

Randomized singular value decomposition:
```python
def power_iteration(AA, Omega, power_iter = 1):
    Y = AA @ Omega
    for q in range(power_iter):
        Y = AA @ (AA.T @ Y)
    Q, _ = np.linalg.qr(Y)
    return Q

def rsvd(mat, Omega):
    A = mat.copy()
    Q = power_iteration(A, Omega)
    B = Q.T @ A
    u_tilde, s, v = np.linalg.svd(B, full_matrices = 0)
    u = Q @ u_tilde
    return u, s, v
```

Temporal graph Fourier transform:
```python
def TGFT():

  return x
```

Conjugate gradient method:
```python
def CG():

  return x
```


```python
def cal_graph_operator(L):
    eigenvalues,eigenvectors = np.linalg.eigh(L)
    inds = np.argsort(eigenvalues)
    U = eigenvectors[:,inds]
    
    return U

def GFT(tensor, U):
    return np.einsum('kt, ijk -> ijt', U, tensor)  #mode-3 product

def iGFT(tensor, U):
    return np.einsum('kt, ijt -> ijk', U, tensor)  #mode-3 product(transpose)
```

basic tensor operations:
```python
def TensorFromMat(mat,dim):
    #Construct a 3D tensor from a matrix
    days_slice = [(start_i,start_i + dim[0]) for start_i in list(range(0,dim[0]*dim[2],dim[0]))]
    array_list = []
    for day_slice in days_slice:
        start_i,end_i = day_slice[0],day_slice[1]
        array_slice = mat[start_i:end_i,:]
        array_list.append(array_slice)
        tensor3d = np.array(np.stack(array_list,axis = 0))
        tensor3d = np.moveaxis(tensor3d,0,-1)
        
    return tensor3d

    
def Tensor2Mat(tensor):
    #convert a tensor into a matrix by flattening the 'day' mode to 'time interval'.
    for k in range(np.shape(tensor)[-1]):
        if k == 0:
            stacked = np.vstack(tensor[:,:,k])
        else:
            stacked = np.vstack((stacked,tensor[:,:,k]))
    return stacked


def construct_Laplacian(adj):
    degree = np.diag(np.sum(adj,axis=1))
    temp = degree-adj
    if np.allclose(temp,temp.transpose()):
        Lap = temp.copy()
    else:
        print('Error Construction')
        Lap = None
    return Lap
```

```python
def tsvt_gft(tensor, Ug, ta,Omg,is_rsvd):
    dim = tensor.shape
    X = np.zeros(dim)
    tensor = GFT(tensor, Ug)
    for t in range(dim[2]):
        if is_rsvd==True:
            u, s, v = rsvd(tensor[:, :, t].T, Omg)
            r = len(np.where(s > ta)[0])
            if r >= 1:
                s = s[: r]
                s[: r] = s[: r] - ta
                X[:, :, t] = (u[:, :r] @ np.diag(s) @ v[:r, :]).T
        else:
            u, s, v = np.linalg.svd(tensor[:, :, t], full_matrices = False)
            r = len(np.where(s > ta)[0])
            if r >= 1:
                s = s[: r]
                s[: r] = s[: r] - ta
                X[:, :, t] = u[:, : r] @ np.diag(s) @ v[: r, :]
    return iGFT(X, Ug)
```


## Example

Performing kriging on the California sensor network:
<p align="center">
<img align="middle" src="graphics/Fig9_.png" alt="fig9" width="800">
</p>

## Reference

  >Please cite our paper if this repo helps your research.

#### Cited as:
bibtex:

```
@misc{nie2023correlating,
      title={Correlating sparse sensing for large-scale traffic speed estimation: A Laplacian-enhanced low-rank tensor kriging approach}, 
      author={Tong Nie and Guoyang Qin and Yunpeng Wang and Jian Sun},
      year={2023},
      eprint={2210.11780},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```


License
--------------

This work is released under the MIT license.
