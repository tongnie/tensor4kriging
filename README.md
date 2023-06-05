## Spatiotemporal graph-embedded low-rank tensor learning for large-scale traffic speed kriging
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)


## To be completed
> This is the code repository for our preprint(manuscript) 'Correlating sparse sensing for network-wide traffic speed estimation: An integrated graph tensor-based kriging approach' submitted to Transportation Research Part C, which is publicly available after publication.


## Datasets
We adopt the large-scale PeMS-4W data to demonstrate how to implement LETC model to perform kriging with missing data imputation.
- **PeMS-4W**: Large-scale traffic speed data measured by 11160 static sensors from the [performance measurement system](https://pems.dot.ca.gov/) in California. The first four weeks of loop speed data with a 5-min window is pre-processed and available at [zenodo](https://zenodo.org/record/3939793).


## Model implementations
Our model is based on a consise NumPy implementation on CPU devices, which is also applicable with CuPy on a GPU device.

```python
import numpy as np
import pandas as pd

tensor = np.load('tensor.npz')['arr_0']

```

## Examples

## References

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
