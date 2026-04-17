import pandas as pd
from src.database.connection import create_session_factory, DEFAULT_DATABASE_URL
from src.features import build_training_dataset
from src.models.training import prepare_training_matrices, train_baseline_regressor
from sklearn.metrics import r2_score
import numpy as np

sf = create_session_factory(DEFAULT_DATABASE_URL)
for m in [10, 50, 100, 200]:
    try:
        ds = build_training_dataset(sf, '2024-25', min_minutes=m)
        arts = train_baseline_regressor(ds)
        print(f"min_minutes={m}: Test R2 = {arts.metrics['test_r2']:.4f}")
    except Exception as e:
        print(f"min_minutes={m}: Error {e}")
