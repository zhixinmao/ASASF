## ASA South Florida Student Data Challenge – NHANES HDL Prediction

This repo trains a leakage-safe ML pipeline to predict `LBDHDD_outcome` and generates `pred.csv` for submission.

### Files
- `prediction.py`: main script (train + predict)
- `README.md`: instructions

### Data
Place:
- `train_dat.csv`
- `test_dat.csv`
in the project root (or `data/` folder—update script accordingly).

### Run
python prediction.py

### Output
Writes `pred.csv` with one column `pred` in the same row order as `test_dat.csv`.

### Reproducibility
Fixed random seed = 7 (CV split and model tuning).
