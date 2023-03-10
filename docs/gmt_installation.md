In case, you would like to run the gmt scripts withing this 
repository you can create a separate conda environment using the
commands bellow:
```bash
conda create --name gmt6
conda activate gmt6
conda config --prepend channels conda-forge
conda install python=3.9 gmt
```