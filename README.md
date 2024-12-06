# AlfaBattle

LSTM network for mcc-code and mcc category code prediction.

## Project Structure

```
yourproject/
│
├── data/
│   └── # Data files
│
├── notebooks/
│   └── mcc_lstm.ipynb  # Jupyter notebooks
│
├── scripts/
│   ├── cat_cat_prediction.py  # Script for category-category prediction
│   ├── cat_mcc_prediction.py  # Script for category-MCC prediction
│   └── mcc_mcc_prediction.py  # Script for MCC-MCC prediction
│
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Install dependencies

```
poetry install
```

## Run the scripts

Predict the next `mcc` code based on previous `mcc` values
```
python scripts/mcc_mcc_prediction.py
```

Predict the next `mcc` code based on previous `mcc_category` values
```
python scripts/cat_mcc_prediction.py
```

Predict the next `mcc_category` based on previous `mcc_category` values
```
python scripts/cat_cat_prediction.py
```

