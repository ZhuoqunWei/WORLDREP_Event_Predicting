# Dataset Download Link:
https://huggingface.co/datasets/Daehoon/WORLDREP/resolve/main/worldrep_dataset_v2.csv
pip install lightgbm==4.3.0 shap tqdm

# Execution Order
1. create venv virtual environment： python3 -m venv .venv
2. download dataset:    python download_dataset.py
3. preprocessing & baseline model:  python baseline_pipeline.py --csv data/worldrep_dataset_v2.csv --outdir artifacts/  