import pandas as pd
import numpy as np
import xgboost as xgb
import config
from prepare_data import add_date_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE 

# load data
df = pd.read_csv(config.PATH, header=None, names=['date', 'price'])
df.to_csv('../datasets/raw/data_raw.csv', index= False)

# prepare data
df = add_date_features(df, 'date')
df.to_csv('../datasets/processed/data_prepared.csv', index= False)

# train test split
X = df.drop([config.TARGET], axis=1)
y = df[config.TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# train model
model = xgb.XGBRegressor(n_estimators=1000)
model.fit(X_train, y_train)
model.save_model("../models/model.json")

# predict target
y_pred = model.predict(X_test) 
  
# RMSE computation 
rmse = np.sqrt(MSE(y_test, y_pred)) 
print("RMSE : % f" %(rmse))
