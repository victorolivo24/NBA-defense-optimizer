import pandas as pd
from src.database.connection import create_session_factory, DEFAULT_DATABASE_URL
from src.features import build_training_dataset
from src.models.training import prepare_training_matrices
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

sf = create_session_factory(DEFAULT_DATABASE_URL)
ds = build_training_dataset(sf, '2024-25', min_minutes=10)
features, target, weights = prepare_training_matrices(ds, 'defensive_rating_target')
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(x_train, y_train)

pred_train = model.predict(x_train)
pred_test = model.predict(x_test)
print(f"XGB Train R2: {r2_score(y_train, pred_train):.4f}")
print(f"XGB Test R2: {r2_score(y_test, pred_test):.4f}")
