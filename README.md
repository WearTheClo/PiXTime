# PiXTime: A Model for Federated Time Series Forecasting with Heterogeneous Data Structures Across Nodes

This project is the official implements of the PiXTime time series forecasting model via PyTorch and NumPy.

## Setup

### Prerequisites
- Install PyTorch and NumPy:
```bash
pip install torch numpy
```

### Dataset
To comply with double-blind review requirements, direct dataset links are not provided during the review phase. The datasets used in our experiments are widely adopted in time series forecasting research and can be easily obtained from public open-source platforms.

Place the obtained dataset files in the `dataset` folder to run the experiments.

## Usage

To run the experiments, simply execute the corresponding `.sh` batch file for each model. This will reproduce the experimental results as shown below:

```bash
# Example ./pixtime_run.sh
./model_run.sh
```

## Experimental Results

The results obtained from running the models will be displayed in the format shown below:

<img width="1778" height="524" alt="QQ_1767686285188" src="https://github.com/user-attachments/assets/79f2e094-abdd-48d2-8a9b-e7af220dd527" />

## Notes
- Ensure all dependencies are installed before running the scripts
- Verify the dataset is correctly placed in the `dataset` folder
- Each model has its own batch file for easy execution
