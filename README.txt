# 1DCNN-SMamba with VMD-CEEMDAN for PV Power Forecasting

This repository contains the custom codebase for our paper published/submitted in *Scientific Reports*.

## Dependencies
To reproduce the experiment, please ensure you have the following Python packages installed:
`pip install pandas numpy matplotlib dtaidistance kmedoids scikit-learn vmdpy EMD-signal torch`

## Usage
1. Place the dataset `data.csv` in the same directory as the script.
2. Run `VMD-CEEMDAN-1DCNN-S-Mamba.py`.
3. The script will automatically perform clustering, decomposition, train the PyTorch model with Early Stopping (max 100 epochs), and save the best weights into the `/saved_models` folder.