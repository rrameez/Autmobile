from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

from pandas import read_csv
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

filename = "C:\\Auto1\\Auto1.csv"
auto1 = read_csv(filename, sep=",",skipinitialspace=True, na_values = ["['']","?", None])

#drop all nan rows!
auto1 = auto1.dropna(axis=0, how = "any")
backup = auto1

#List of categorial variables
categorical = ["make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type","num-of-cylinders","fuel-system"]

#toDrop = ["symboling", "engine-location","fuel-system", "price"]
toDrop = ["symboling", "price"]

price = auto1["price"]

for item in categorical:
    #not changing num-of-cylinders into one-hot encoding as this is better as an ordinal variable
    if item == "num-of-cylinders":
        e = preprocessing.LabelEncoder()
        auto1[item] = e.fit_transform(auto1[item])
        continue

    one_hot = pd.get_dummies(auto1[item])
    auto1 = auto1.drop(item,axis=1)
    auto1 = auto1.join(one_hot)

for item in toDrop:
    auto1 = auto1.drop(item, axis=1)

auto1 = auto1.join(price)

X1 = auto1.values

X = np.array(X1[:,0:58])
Y = np.array(X1[:,59])
Y = np.reshape(Y, (-1, 1))
print("xy", X.shape, Y.shape)
validation_size = 0.3
seed = 1

print("~~~~~~~~~~~~~~~~USE CASE 1(a). To check which features are important to provide to users for initial appraisal")

check = backup.values
col_idx= [18,19,20,21,22,23,24]
RFEX = np.array(check[:,col_idx])
RFEy = np.array(check[:,25])
RFEy = np.reshape(RFEy, (-1, 1))
validation_size = 0.3

subsetNames = ["bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg"]

estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(RFEX, RFEy)

for i in range(len(selector.ranking_)):
    if selector.ranking_[i] == 1:
        print(subsetNames[i]," can be asked of the user initially")

print("\n Given that MPG is available on most car dashboards, that can be selected!")

print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\n Now Onto USE CASE 1 (b) ")

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)

param_grid = {"alpha": [-1, -10, 1, 10, 100],
              "kernel": ['linear']}
sv = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
pred = sv.fit(X_train,Y_train).predict(X_validation)

#print("new score",cross_val_score(sv, X_validation, Y_validation, scoring='neg_mean_squared_error'))

print("The R-square measure for Kernel Ridge on Test set is", sv.score(X_validation,Y_validation))

print("Mean Absolute Error is ", mean_absolute_error(Y_validation,pred))

print("\n Now checking a host of models, cross validated")

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(("KR", KernelRidge()))


num_folds = 10
seed = 1
scoring = 'mean_absolute_error'

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = (name, cv_results.mean(), cv_results.std())
    print(msg)


results = []
names = []

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',
Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(name, cv_results.mean(), cv_results.std())

results = []
names = []

ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',
AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',
GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',
RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',
ExtraTreesRegressor())])))

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~ ENSEMBLE METHODS ~~~~~~~~~~~~~~~~~~~~~")

for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(name, cv_results.mean(), cv_results.std())

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
param_grid = dict(n_estimators=np.array([1,50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(mean, stdev, param)