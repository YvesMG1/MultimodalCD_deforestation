# Comparing fusion techniques for Sentinel-2 and Sentinel-1 in deforestation change detection

This repository contains the implementation of my MSc Thesis at ETH Zürich. In this project, we compare various deep learning architectures to improve deforestation change detection by fusing multimodal data from Sentinel-2 (optical) and Sentinel-1 (radar) imagery.

To replicate our results or evaluate on new data, follow the instructions below.

## 1. Data

The final dataset is available on: [Google Drive](https://drive.google.com/drive/u/0/folders/19ilDM-GLARZzeTcCwYvBUlDq9ltgHl4I).

## 2. Training

You can train the models by running:

```bash
python src/training/main.py --config /models/configs/config_file.yaml
```

### Example usage:

```bash
python src/training/main.py --config /models/configs/config_ukr_cd.yaml
```

Make sure to specify the root path to your data folder in the configuration file.

## 3. Evaluation

All pretrained models are stored in `models/fitted_models`.

For evaluation and to visualize results, follow the instructions in:

```bash
notebooks/model_evaluation.ipynb
```

## Dependencies

All dependencies are listed in the `environment.yml` file.


