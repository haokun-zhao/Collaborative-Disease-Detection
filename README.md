# Collaborative-Disease-Detection
This is the repo for paper "Collaborative Disease Detection"

## Set up
First create an empty conda environment with Python 3 (reference version: Python 3.10.18) and `pip install` the `requirements.txt` file in this directory.
```sh
conda create -n cdd python=3.10.18
conda activate cdd
pip install -r requirements.txt
```

## Run the code
```sh
cd CDD
python main.py
```

## Figures and Results
Fig. 1 illustrutes the core concepts of high-order connectivity:
<img src="./Figure/high order connectivity.png" width="100%"></img>

Table 1 lists the performance comparison on MIMIC-IV:
<img src="./Figure/performance comparison.png" width="100%"></img>
