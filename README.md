# Edmonton Housing Price Model

PyTorch model for predicting Edmonton residential assessed values using property attributes, LRT proximity, school catchments, and recreation facilities. Includes a map-based prototype with address lookup.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `Property_Assessment_Data.csv` from the [City of Edmonton Open Data Portal](https://data.edmonton.ca/) and place it in the `data/` folder before training.

## Train

```bash
python housing_price_pytorch.py --sample-size 0
```

Use `--sample-size 500000` for a faster run on a subset.

## Prototype UI

```bash
uvicorn prototype.server:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080
