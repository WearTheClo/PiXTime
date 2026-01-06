# PiXTime: A Model for Federated Time Series Forecasting with Heterogeneous Data Structures Across Nodes

This project is the official implements of the PiXTime time series forecasting model via PyTorch and NumPy.

## Setup

### Prerequisites
- Install PyTorch and NumPy:
```bash
pip install torch numpy
```

### Dataset
1. Download the dataset from [[link]](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
2. Place the dataset in the `dataset` folder

**Acknowledgement**: We would like to thank the Autoformer project team for organizing and providing the dataset. Their project is available at: https://github.com/thuml/Autoformer?tab=readme-ov-file

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
