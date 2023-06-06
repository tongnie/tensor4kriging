## Spatiotemporal graph-embedded low-rank tensor learning for large-scale traffic speed kriging
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)


## To be completed
> This is the code repository for our preprint(manuscript) 'Correlating sparse sensing for network-wide traffic speed estimation: An integrated graph tensor-based kriging approach' submitted to Transportation Research Part C, which is publicly available after publication.


## Motivation
Traffic speed is central to characterizing the fluidity of the road network. Many transportation applications rely
on it, such as real-time navigation, dynamic route planning, and congestion management. However, due to sparse
deployment of static sensors or low penetration of mobile sensors, speeds detected are incomplete and far from
network-wide use. In addition, sensors are prone to error or missing data due to various kinds of reasons, speeds
from these sensors can become highly noisy. These drawbacks call for effective techniques to recover credible
estimates from the incomplete data. In this repository we demonstrated a **L**aplacian-**e**nhanced low-rank **t**ensor **c**ompletion (LETC) framework featuring both low-rankness and multi-dimensional correlations for large-scale traffic speed kriging under limited observations. 


## Dataset
We adopt the large-scale PeMS-4W data to demonstrate how to implement LETC model to perform kriging with missing data imputation.
- **PeMS-4W**: Large-scale traffic speed data measured by 11160 static sensors from the [performance measurement system](https://pems.dot.ca.gov/) in California. The first four weeks of loop speed data with a 5-min window is pre-processed and available at [zenodo](https://zenodo.org/record/3939793).

Load graph information:

'''python
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
'''


## Model implementation
Our model is based on a consise NumPy implementation on CPU devices, which is also applicable with CuPy on a GPU device. Some key operations are discussed as below:

Prepare tensors:

```python
import numpy as np
import pandas as pd

tensor = np.load('xxx.npz')

```

Randomized tensor singular value decomposition:
```python
def r_tsvd():

  return x
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

## Example

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
