# train_Harsh.py
# this script trains the lightgbm model on car auction data
# saves model and encoders as pkl files for the agent to use later

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


# loading the dataset
df = pd.read_csv("car_auction_train.csv")
print("shape of data:", df.shape)


# ---- DATA CLEANING ----

# removing prices that dont make sense
# cars below 500 are probably data entry errors or salvage parts
# above 150k are ultra luxury outliers, very rare and mess up the model
before = len(df)
df = df[(df['sellingprice'] >= 500) & (df['sellingprice'] <= 150000)]
print(f"removed {before - len(df)} rows based on price filter")

# odometer above 300k is probably sensor reset/glitch on old cars
# saw values like 999999 in the data which is clearly wrong
df = df[df['odometer'] <= 300000]

# filling missing numerical values with median
# using median because odometer has outliers that would skew the mean
df['odometer'].fillna(df['odometer'].median(), inplace=True)
df['condition'].fillna(df['condition'].median(), inplace=True)

# for text columns just filling with Unknown
# didnt use mode because Unknown as a separate category gives model more info
CAT_COLS = ['make', 'model', 'trim', 'body', 'transmission', 'color', 'interior', 'state']
for col in CAT_COLS:
    df[col] = df[col].fillna('Unknown')

print("rows after cleaning:", len(df))


# ---- FEATURE ENGINEERING ----

CURRENT_YEAR = 2015  # max year in this dataset

# car age is more meaningful than raw year for depreciation patterns
df['car_age'] = CURRENT_YEAR - df['year']
df['car_age'] = df['car_age'].clip(lower=0)  # some entries had future years, clip to 0

# how hard was the car driven each year
# this is better than raw odometer for newer cars
# eg 80k miles on a 3yr old car is very different from 80k on a 10yr old car
df['odo_per_year'] = df['odometer'] / df['car_age'].replace(0, 0.5)

# older cars in bad condition are much worse than just old OR just bad condition
# this interaction captures that combined effect
df['condition_x_age'] = df['condition'] * df['car_age']

# cars with low mileage AND good condition get a premium
# buyers pay extra for "almost new" cars
df['low_odo_good_cond'] = (df['odometer'] < 30000).astype(int) * df['condition']


# ---- LABEL ENCODING ----
# converting text columns to numbers, saving encoders for later use in agent
# have to save them because agent gets one car at a time, cant refit

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

FEATURE_COLS = (
    ['year', 'car_age', 'odometer', 'odo_per_year',
     'condition', 'condition_x_age', 'low_odo_good_cond']
    + [c + '_enc' for c in CAT_COLS]
)

X = df[FEATURE_COLS].values
y = df['sellingprice'].values

# 85/15 split, we have 4.5 lakh rows so 15% validation is still huge
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"train size: {len(X_train)}  val size: {len(X_val)}")


# ---- HYPERPARAMETER TUNING WITH OPTUNA ----
# tried gridsearch first but too slow, optuna is smarter and faster
# it learns from previous trials and searches in better directions

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 63, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # early stopping so it doesnt waste time if model stops improving
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)]
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


print("\nstarting optuna search, 30 trials...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=False)

best_params = study.best_params
print(f"best val RMSE found: ${study.best_value:,.0f}")
print(f"best params: {best_params}")


# ---- TRAINING FINAL MODEL ----
# now training on full data (not just 85%) using the best params found above
# more data = better model

print("\ntraining final model on all data...")

best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['verbose'] = -1

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X, y)

# checking val rmse just for reporting (slightly optimistic since val data is included now)
val_preds = final_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"final model RMSE: ${val_rmse:,.0f}")


# ---- SAVING ----
# saving model with feature list and current year so agent has everything it needs
with open('model_Harsh.pkl', 'wb') as f:
    pickle.dump({
        'model': final_model,
        'features': FEATURE_COLS,
        'current_year': CURRENT_YEAR
    }, f)

# saving encoders separately, agent needs these to convert make/model/etc to numbers
with open('encoders_Harsh.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("\ndone! saved model_Harsh.pkl and encoders_Harsh.pkl")
