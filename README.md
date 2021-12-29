
# <div align="center">House Price prediction</div>
<div align="center">
  <img src = "https://media.giphy.com/media/UqqVRaP8y4uo1GNxbN/giphy.gif" width="65%">
</div>

## Objectives:
Predict sales prices and practice feature engineering, RFs, and gradient boosting
## Dataset:
 [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### Data Fields:
The Dataset has With 79 explanatory variables describing (almost) every aspect of residential homes in Ames and Iowa.
- **SalePrice**: the property's sale price in dollars. This is the target variable that you're trying to predict.
- **MSSubClass**: The building class
- **MSZoning**: The general zoning classification
- **LotFrontage**: Linear feet of street connected to property
- **LotArea**: Lot size in square feet
- **Street**: Type of road access
- **Alley**: Type of alley access
- **LotShape**: General shape of property
- **LandContour**: Flatness of the property
- **Utilities**: Type of utilities available
- **LotConfig**: Lot configuration
- **LandSlope**: Slope of property
- **Neighborhood**: Physical locations within Ames city limits
- **Condition1**: Proximity to main road or railroad
- **Condition2**: Proximity to main road or railroad (if a second is present)
- **BldgType**: Type of dwelling
- **HouseStyle**: Style of dwelling
- **OverallQual**: Overall material and finish quality
- **OverallCond**: Overall condition rating
- **YearBuilt**: Original construction date
- **YearRemodAdd**: Remodel date
- **RoofStyle**: Type of roof
- **RoofMatl**: Roof material
- **Exterior1st**: Exterior covering on house
- **Exterior2nd**: Exterior covering on house (if more than one material)
- **MasVnrType**: Masonry veneer type
- **MasVnrArea**: Masonry veneer area in square feet
- **ExterQual**: Exterior material quality
- **ExterCond**: Present condition of the material on the exterior
- **Foundation**: Type of foundation
- **BsmtQual**: Height of the basement
- **BsmtCond**: General condition of the basement
- **BsmtExposure**: Walkout or garden level basement walls
- **BsmtFinType1**: Quality of basement finished area
- **BsmtFinSF1**: Type 1 finished square feet
- **BsmtFinType2**: Quality of second finished area (if present)
- **BsmtFinSF2**: Type 2 finished square feet
- **BsmtUnfSF**: Unfinished square feet of basement area
- **TotalBsmtSF**: Total square feet of basement area
- **Heating**: Type of heating
- **HeatingQC**: Heating quality and condition
- **CentralAir**: Central air conditioning
- **Electrical**: Electrical system
- **1stFlrSF**: First Floor square feet
- **2ndFlrSF**: Second floor square feet
- **LowQualFinSF**: Low quality finished square feet (all floors)
- **GrLivArea**: Above grade (ground) living area square feet
- **BsmtFullBath**: Basement full bathrooms
- **BsmtHalfBath**: Basement half bathrooms
- **FullBath**: Full bathrooms above grade
- **HalfBath**: Half baths above grade
- **Bedroom**: Number of bedrooms above basement level
- **Kitchen**: Number of kitchens
- **KitchenQual**: Kitchen quality
- **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms)
- **Functional**: Home functionality rating
- **Fireplaces**: Number of fireplaces
- **FireplaceQu**: Fireplace quality
- **GarageType**: Garage location
- **GarageYrBlt**: Year garage was built
- **GarageFinish**: Interior finish of the garage
- **GarageCars**: Size of garage in car capacity
- **GarageArea**: Size of garage in square feet
- **GarageQual**: Garage quality
- **GarageCond**: Garage condition
- **PavedDrive**: Paved driveway
- **WoodDeckSF**: Wood deck area in square feet
- **OpenPorchSF**: Open porch area in square feet
- **EnclosedPorch**: Enclosed porch area in square feet
- **3SsnPorch**: Three season porch area in square feet
- **ScreenPorch**: Screen porch area in square feet
- **PoolArea**: Pool area in square feet
- **PoolQC**: Pool quality
- **Fence**: Fence quality
- **MiscFeature**: Miscellaneous feature not covered in other categories
- **MiscVal**: $Value of miscellaneous feature
- **MoSold**: Month Sold
- **YrSold**: Year Sold
- **SaleType**: Type of sale
- **SaleCondition**: Condition of sale

## Implementation:

**Libraries:** `sklearn` `Matplotlib` `pandas` `seaborn` `NumPy` `Scipy` `XGBoost` `lightgbm`



## Few glimpses of EDA:
### 1. Discrete Variables:
Here, we have analyzed each discrete variable individually and make decisions based on correlation with SalePrice.<br>
I have included Data visualizatio  for some these variables, for complete analysis hace alook at : [Notebook](https://github.com/Pradnya1208/House-prices-prediction/blob/main/House%20prices%20prediction.ipynb)

#### Number of Bathrooms
> ![Barhs](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/baths.PNG?raw=true)

#### Bedrooms above grade
> ![Bedrooms](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/bedrooms.PNG?raw=true)

#### Kitchen above grade
> ![Kitchen](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/kitchen.PNG?raw=true)

#### Toatal Rooms
> ![Rooms](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/totalrooms.PNG?raw=true)

#### Fire Places
> ![Fireplace](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/Fireplaces.PNG?raw=true)

#### Size of Garage in Car Capacity
> ![Garage](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/gragesize.PNG?raw=true)

#### Remodel Date
> ![Remodelling](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/remodellingyear.PNG?raw=true)

#### Original Construction Date
> ![construction](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/yearbuilt.PNG?raw=true)

#### Year Sold
> ![sold](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/yearsold.PNG?raw=true)

### 2. Continuous Variables:
Here we are going to analyze correlation of each feature with SalePrice, see skewness for linear and boxcox transformations and apply the better one. If skewness continues high, we will bin the variable into categories or flag (0 and 1). And if there are missing values, we will drop the column.

#### Linear feet of street connected to property:
> ![Loft frontage](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/Loft_continuous.PNG?raw=true)

#### Area of Lot:
> ![Lot](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/lot_cont.PNG?raw=true)

#### Masonry veneer area in square feet:
> ![veneer](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/veneer_cont.PNG?raw=true)

#### Basement:
> ![base](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/basement.PNG?raw=true)

Transforming this feature into categories
> ![Category](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/basement_categorical.PNG?raw=true)
> ![corr](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/basement_after-categorical.PNG?raw=true)
since there is no correlation, we can flag this feature.
> ![flagged](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/flagged.PNG?raw=true)

We have transformed several continuous features in the same way, for more details have a look at the [Notebook](https://github.com/Pradnya1208/House-prices-prediction/blob/main/House%20prices%20prediction.ipynb)

### 3.Ordinal Varibles:
Here, we will first change the strings by integers, because the variables are ordinal, we can't get dummies unless the correlation with SalePrice is very low. Correlation, missing values are taken into consideration.

#### Shape of the Property:
> ![Lot](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/lotshape_ordinal.PNG?raw=true)

#### Type of utilities available:
> ![Utilities](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/utility_type.PNG?raw=true)

#### Slope of Property:
> ![slope](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/property_slope.PNG?raw=true)

#### If Central air conditioning is present.:
> ![AC](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/central%20AC_ordinal.PNG?raw=true)

#### Electrical System:
> ![Electrical](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/electrical_system_ordinal.PNG?raw=true)

### 4. Nominal Variales:
Here, we will analyze correlation with the boxplots and missing values. Clustering information when necessary from categories, decisions will be made to drop, flag or keep the column.

#### Road Access:
> ![road access](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/road_access_ordinal.PNG?raw=true)

#### Roof Style:
> ![roof](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/roof.PNG?raw=true)

## Model Training and Evaluation:
In this section, we are going to get dummies for categorical variables, split train and test sets, analyze skewness if yet present for both sets, scale the data (Robust is better for outliers) and, finally, train the model for
* Lasso
* ElasticNet
* Kernel Ridge
* Gradient Boosting Regressor
* XGBoost
* Light Gradient Boosting

We've obtained the tuning parameters from GridSearchCV
before training the model we have done following steps
- Analyzing skewness and replacing column with the Boxcox transformation
- Scaling with RobustScaler
- We've used log transformation for Target variable

**Lasso**
```
lasso = Lasso(alpha= 0.0005)
```

**Elastic Net**
```
elastic = ElasticNet(alpha=0.0005, l1_ratio=.9)
```

**Kernel Ridge**
```
k_ridge = KernelRidge(alpha=0.1, coef0=2.5, degree=3, gamma=None, kernel='polynomial',kernel_params=None)
```

**Gradient Boost**
```
g_boost = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse',
                          init=None, learning_rate=0.01, loss='huber',
                          max_depth=3, max_features=18, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=3, min_samples_split=5,
                          min_weight_fraction_leaf=0.0, n_estimators=3300,
                          n_iter_no_change=None,random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
```

**XGBoost**
```
xg_boost = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1, missing=None, n_estimators=3300,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=0.3, verbosity=1)
```
**Light GBM**
```
lgbm = LGBMRegressor(bagging_fraction=0.6, bagging_freq=9, bagging_seed=14,
              boosting_type='gbdt', class_weight=None, colsample_bytree=0,
              feature_fraction=0.1, feature_fraction_seed=1,
              importance_type='split', learning_rate=0.01, max_bin=65,
              max_depth=4, min_child_samples=20, min_child_weight=0.001,
              min_data_in_leaf=5, min_split_gain=0.0, min_sum_hessian_in_leaf=5,
              n_estimators=3300, n_jobs=-1, num_leaves=7,
              objective='regression', random_state=None, reg_alpha=0.2,
              reg_lambda=0.1, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0)
```

**Stacking**
![Stacking](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/stacking.PNG?raw=true)

```
stacking = StackingRegressor(regressors=(elastic, g_boost, k_ridge),
                             meta_regressor = lasso)

param_grid = {} 

stack = GridSearchCV(stacking, 
                   param_grid = param_grid,
                   cv = 10, 
                   scoring = "neg_mean_squared_error",
                   n_jobs = 5, 
                   verbose = 1)

stack.fit(X_train,Y_train)
```
```
best score: -0.01181972103207294
```

**Voting Regressor**
A voting regressor is an ensemble meta-estimator that fits base regressors each on the whole dataset. It, then, averages the individual predictions to form a final prediction.

```
voting = VotingRegressor(estimators=[('xgboost', xg_boost), 
                                     ('lgbm', lgbm),
                                     ('stacking', stacking)])

v_param_grid = {} # tuning voting parameter

gsV = GridSearchCV(voting, 
                   param_grid = v_param_grid,
                   cv = 10, 
                   scoring = "neg_mean_squared_error",
                   n_jobs = 5, 
                   verbose = 1)

gsV.fit(X_train,Y_train)
```
```
best score:-0.011459485092086497
```


### Optimizations

For Hyperparameter Tuning we've used GridSearchCv.

### Lessons Learned

`Data Imputation`
`Handling Outliers`
`Feature Engineering`
`Advanced regression techniques`
`Stacking`
`Voting`

### References:

- [Voting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)
- [Stacking](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor)
- [XGBoost](https://dask-ml.readthedocs.io/en/stable/modules/generated/dask_ml.xgboost.XGBRegressor.html)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [Kernel Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
- [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [Scaling](http://benalexkeen.com/feature-scaling-with-scikit-learn/)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner


[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]


