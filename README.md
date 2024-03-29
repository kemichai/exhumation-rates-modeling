## Exhumation rates and thermal structure for the central Southern Alps, New Zealand

Set of python codes for reproducing the results from the manuscript entitled
"Crustal thermal structure and exhumation rates near the central Alpine Fault,
New Zealand" published in G^3 in 2020 [link](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020GC008972).

In particular exhumation rates estimates are calculated along the length of
the Alpine Fault using a steady state 1-D thermal model assuming steady
using a number of boundary conditions and two separate datasets:

* Sesmicity observations (earthquake hypocentral depths)
* Thermochronological ages

Author: [`Konstantinos Michailos`](https://github.com/kemichai) (Developer and Maintainer) 

Installation:

1 ) Clone repository:
```bash
git clone https://github.com/kemichai/exhumation_rates_modeling.git
cd exhumation_rates_modeling
```
2 ) Create a conda environment:
The simplest way to run the Python codes is using anaconda and a virtual environment [link to install conda](https://docs.conda.io/en/latest/miniconda.html).

Once you have installed conda, create a new environment with the following dependencies using:
```bash
conda config --add channels conda-forge
conda create -n sa_exhum python=3.7 pip obspy=1.1.0 matplotlib numpy pandas pyproj shapely basemap
conda activate sa_exhum
conda install gmt -c conda-forge/label/dev
```

3 )
To install the functions and make them available everywhere on your machine (within your environment)
type the following:

```bash 
pip install .
```

How to run the code
--------------------
Having installed the conda environment you should be able to create the thermal model 
we present in the publication by running the following command:
```python
python run_model.py
```
All the parameters for tuning the model can be found withing the 
file `run_model.py`.

To reproduce the manuscript plots please refer to the directory called 
manuscript plots.

Note
------------
Codes are designed to reproduce our results in the current publication.
For different applications the codes will need to be modified. 

(WIP)

TODO: fix issue with Basemap...
