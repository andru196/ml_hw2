import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error


def comp_mse(df, features, target):
    kf = KFold(n_splits=5, shuffle=True) #random_state=42)
    
    msle_list = []
    mse_list = []
    for i, (train_index, val_index) in enumerate(kf.split(df)):    
        train_part = df.iloc[train_index, : ]
        val_part = df.iloc[val_index, :]
        model = SGDRegressor()
        model.fit(X=train_part[features].fillna(0), y = train_part[target])

        val_predication = model.predict(val_part[features].fillna(0)).clip(0, 100000000000)

        mse = mean_squared_error(val_predication, val_part[target])
        msle = mean_squared_log_error(val_predication, val_part[target])

        pirce_mse = mean_squared_error(val_predication * val_part['full_sq'], val_part['price_doc'])
        pirce_msle = mean_squared_log_error(val_predication * val_part['full_sq'], val_part['price_doc'])

        print(f'Fold full {i}: msle {pirce_msle}, mse {pirce_mse}')
        msle_list.append(pirce_msle)
        mse_list.append(pirce_mse)
        
    print(f'MSLE average = {np.mean(msle_list)}, std = {np.std(msle_list)}')
    
    
