import numpy as np
from lightgbm import LGBMRegressor
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


#Separate target from features
df = pd.read_csv('content/car_prices.csv')
cols_to_drop = ['transmission', 'vin', 'state', 'condition', 'color', 'interior', 'mmr', 'saledate', 'seller']
df = df.drop(cols_to_drop, axis=1)

#drop rows with null values
df = df.dropna()

X = df.drop(['sellingprice'], axis=1)
y = df['sellingprice']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X[categorical_features] = X[categorical_features].apply(lambda x: x.astype(str).str.lower().str.replace(' ', ''))
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
)

preprocessor.set_output(transform="pandas")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing to training and testing data
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)



# Define the Final Production Model, it was already tweaked for best results
final_model = LGBMRegressor(
    objective='regression',
    random_state=42,
    n_estimators=5000,          
    learning_rate=0.02,         
    num_leaves=150,             
    max_depth=-1,
    min_child_samples=5,        
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_model)
])

print("Training final production pipeline")
pipeline.fit(X_train_raw, y_train) 
print("Training complete.")

model_to_test = final_model
y_pred = model_to_test.predict(X_test)

#Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#print metrics
print("\nPerformance Metrics for Best LightGBM Model on Test Set:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE):      {mae:.2f}")
print(f"R² Score (Accuracy):            {r2:.4f}")

#show examples
print("\nSample Comparisons (Actual vs Predicted):")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head(10))

model_filename = 'estimate_veh_price.joblib'
joblib.dump(pipeline, model_filename)