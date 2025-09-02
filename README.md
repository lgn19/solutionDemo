# Sound Source Localization Project Description

## Project Overview
This project is used to implement 3D localization prediction of sound sources using SVR regression models based on sound data collected by a microphone array. The project includes three main stages: data generation, feature extraction, and model training & prediction.

## Environment Setup

### Install Required Libraries
Ensure that all dependent libraries required by the project are installed (the specific list of libraries can be supplemented according to the actual code, such as numpy, scikit-learn, etc.).

## Data Generation

### Run the Data Generation Script
```bash
python generate_data.py sound/p225 generated --n_voices 1 --n_outputs 300 --mic_radius 0.32 --n_mics 16
```

### Data Description
- Location of original data: `sound/p225`
- Location of generated simulation data: `generated`
- Parameters of generated data:
  - Number of samples: 300
  - Number of sound sources per sample: 1
  - Number of microphones: 16

### Microphone Array Positions
```python
mic_positions = [
    [1.8, 0.6, 1], [1.8667, 0.6, 1], [1.9333, 0.6, 1], [2, 0.6, 1],
    [2.4, 0.6, 1], [2.4667, 0.6, 1], [2.5333, 0.6, 1], [2.6, 0.6, 1],
    [1.8, 0.6, 2], [1.8667, 0.6, 2], [1.9333, 0.6, 2], [2, 0.6, 2],
    [2.4, 0.6, 2], [2.4667, 0.6, 2], [2.5333, 0.6, 2], [2.6, 0.6, 2]
]
```

### Room Geometry Parameters
```python
corner = np.array([[0, 0], [8, 0], [8, 5], [0, 5]]).T
```

## Feature Extraction

Run the feature extraction script:
```bash
python run_example.py
```

### Function Description
- Collect DP-RTF DOA features from simulation data
- Gather real coordinates of sound sources as samples
- All samples are stored in the `sample` directory

### Output File Description
- `X.npy`: DOA feature data with dimensions 4×number of samples
- `Y.npy`: Real coordinates of sound sources with dimensions 3×number of samples
- Note: Test samples are already included in the `sample` directory

## Model Training and Prediction

Run the SVR regression model:
```bash
python SVR.py
```

### Model Description
Three solutions (solution-I to solution-III) are built using SVR regression.

### Example Output Results

```
[soluton 1] Angle Prediction Metrics:
  Degree1 - RMSE: 1.4707, R²: 0.9956
  Degree2 - RMSE: 1.4487, R²: 0.9959
  Degree3 - RMSE: 1.5519, R²: 0.9954
  Degree4 - RMSE: 2.1385, R²: 0.9911

[soluton 1] 3D Point Prediction from Angles:
  R1 - RMSE: 0.1116, R²: 0.8433
  R2 - RMSE: 0.1352, R²: 0.7795
  X Coordinate - RMSE: 0.0561, R²: 0.9853
  Y Coordinate - RMSE: 0.1101, R²: 0.8330
  Z Coordinate - RMSE: 0.1810, R²: 0.7513
  Mean Spatial Distance Error: 0.1710

------------------------------

[soluton 2] Direct Prediction Metrics:
  R1 - RMSE: 0.0867, R²: 0.9054
  R2 - RMSE: 0.0844, R²: 0.9142
  X Coordinate (Direct) - RMSE: 0.0256, R²: 0.9969
  X Coordinate (Final) - RMSE: 0.0256, R²: 0.9969
  Y Coordinate - RMSE: 0.0971, R²: 0.8701
  Z Coordinate - RMSE: 0.0893, R²: 0.9395
  Mean Spatial Distance Error: 0.1072

------------------------------

[soluton 3] Full 3D Position Prediction:
  X Coordinate - RMSE: 0.0256, R²: 0.9969
  Y Coordinate - RMSE: 0.0874, R²: 0.8948
  Z Coordinate - RMSE: 0.0934, R²: 0.9338
  Mean Spatial Distance Error: 0.1035
```
