需要安装pytorch与numpy；

从[[link]](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)得到数据集，请将数据集放在dataset文件夹内；这里我们需要对整理数据集的Autoformer项目组表达感谢，他们的项目界面为https://github.com/thuml/Autoformer?tab=readme-ov-file

直接运行每个模型对应的.sh批处理文件即可得到如下所示的实验结果：


# Experiment Reproduction Guide

This repository provides scripts to reproduce the experimental results of several time series forecasting models. The dataset preprocessing and organization follow the excellent work from the **Autoformer** project.

## 📦 Dependencies

Before running the experiments, please ensure the following Python packages are installed:

- PyTorch
- NumPy

You can install them via pip:
```bash
pip install torch numpy
