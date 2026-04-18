import pandas as pd
from src.database.connection import create_session_factory, DEFAULT_DATABASE_URL
from src.features import build_training_dataset
from src.models.training import prepare_training_matrices
from src.models.scheme_recommender import LinearSchemeRecommender
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sf = create_session_factory(DEFAULT_DATABASE_URL)
ds = build_training_dataset(sf, '2024-25', min_minutes=10)
features, target, weights = prepare_training_matrices(ds, 'defensive_rating_target')
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

model = LinearSchemeRecommender()
model.fit(x_train, y_train, sample_weight=weights.loc[x_train.index])

pred_train = model.predict(x_train)
pred_test = model.predict(x_test)
print(f"Linear Train R2: {r2_score(y_train, pred_train):.4f}")
print(f"Linear Test R2: {r2_score(y_test, pred_test):.4f}")
