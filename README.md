
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

#### Type of Foundation:
> ![foundation](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/Foundation.PNG?raw=true)

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

```
**Lasso**
lasso = Lasso(alpha= 0.0005)

```

```
**Elastic Net**
elastic = ElasticNet(alpha=0.0005, l1_ratio=.9)
