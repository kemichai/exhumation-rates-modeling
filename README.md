## Exhumation rates and thermal structure for the central Southern Alps, New Zealand

Set of python codes for reproducing the results from the manuscript entitled
"Crustal thermal structure and exhumation rates near the central Alpine Fault,
New Zealand" published in G^3 in 2020 [link](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020GC008972).

In particular exhumation rates estimates are calculated along the length of
the Alpine Fault using a steady state 1-D thermal model assuming steady
using a number of boundary conditions and two separate datasets:

* Sesmicity observations (earthquake hypocentral depths)
* Thermochronological ages

Installation
------------
The codes in this repository are in Python. 
There are different ways to **install Python**. We recommend using Conda 
(e.g., Miniconda). You can download the installation file based on your operating system and install 
Miniconda using the following [link](https://docs.conda.io/en/latest/miniconda.html)

Once you have installed conda, open a terminal (Linux) 
create a new environment with the following dependencies using:
```bash
conda config --add channels conda-forge
conda create -n sa_exhum python=3.7 pip obspy=1.1.0 matplotlib numpy pandas pyproj shapely basemap
conda activate sa_exhum
conda install gmt -c conda-forge/label/dev
```

Clone the repository by typing the following:
```bash
git clone https://github.com/kemichai/exhumation-rates-modeling.git
cd exhumation_rates_modeling
```
To install the functions and make them available everywhere on your machine (within your environment)
type the following:

```bash 
pip install .
```
That should do it!

How to run the codes
--------------------
Having installed the conda environment you should be able to create the thermal model 
we present in the publication by running the following command:
```python
python run_model.py
```
All the parameters for tuning the model can be found and adjusted within the 
file `run_model.py`.

To reproduce the manuscript plots please refer to the directory called 
manuscript plots. There you can find also some GMT scripts for making the 
maps included in the publication. For details on how to get the GMT scripts
working click [here](docs/gmt_installation.md).


Note
------------
Codes are designed to reproduce our results in the current publication.
For different applications the codes will need to be modified. 
