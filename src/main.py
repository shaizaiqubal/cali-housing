import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/housing.csv")

y=dataset["median_house_value"]
x = dataset.drop(columns=["median_house_value", "ocean_proximity"])

x_tmp,x_test,y_tmp,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_val, x_train, y_val, y_train = train_test_split(x_tmp, y_tmp, test_size=0.25, random_state=42)

# med_bedrooms=x_tmp["total_bedrooms"].median()
# x_tmp["total_bedrooms"]=x_tmp["total_bedrooms"].fillna(med_bedrooms)
# x_test["total_bedrooms"]=x_test["total_bedrooms"].fillna(med_bedrooms)

def feature_engineering(x):
    x=x.copy()
    x["rooms_per_household"]=x["total_rooms"]/x["households"]
    x["bedrooms_per_rooms"]=x["total_bedrooms"]/x["total_rooms"]
    x["density"]=x["population"]/x["households"]
    x["coordinates"]=x["latitude"]*x["longitude"]
    x["income_squared"]=x["median_income"]**2
    return x



imputer = SimpleImputer(strategy='median')
imputer.set_output(transform='pandas')
poly=PolynomialFeatures(degree=2,include_bias=False)
poly.set_output(transform="pandas")
pipe = Pipeline([
    ("imputer", imputer),
    ("features", FunctionTransformer(feature_engineering, validate=False)),
    ("poly",poly),
    ("scaler", StandardScaler())
])

x_train_processed=pipe.fit_transform(x_train)
x_val_processed=pipe.transform(x_val)
x_test_processed=pipe.transform(x_test)

# x_train,x_test,y_train,y_test=train_test_split(x_tmp,y_tmp,test_size=0.25,random_state=42)
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.fit_transform(x_test)

# sdgr=SGDRegressor(random_state=42)
# sdgr.fit(x_train_processed,y_train)
# print(f"w:{sdgr.coef_}  b:{sdgr.intercept_}")

alphas=[0.001,0.01,0.1,1.0,10.0,100,1000,10000]
val_mse=[]
for a in alphas:
    ridge = Ridge(alpha=a, random_state=42)
    ridge.fit(x_train_processed, y_train)
    y_pred=ridge.predict(x_val_processed)
    mse=mean_squared_error(y_val,y_pred)
    val_mse.append(mse)

for a,i in zip(alphas,val_mse):
    print(a,        i)

# Automatically select best alpha
best_alpha = alphas[np.argmin(val_mse)]
print(f"Best alpha: {best_alpha} with validation MSE: {min(val_mse):.2f}")

x_final=np.vstack([x_train_processed,x_val_processed])
y_final=np.hstack([y_train,y_val])

ridge_final=Ridge(alpha=best_alpha)
ridge_final.fit(x_final,y_final)
yt_pred=ridge_final.predict(x_train_processed)
y_pred=ridge_final.predict(x_test_processed)

print(f"training mse after ridge validation: {root_mean_squared_error(y_train,yt_pred):.2f} \ntest rmse after ridge validation: {root_mean_squared_error(y_test,y_pred): .2f}")




