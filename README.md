
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

**Number of Bathrooms**
> ![Barhs](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/baths.PNG?raw=true)

**Bedrooms above grade**
> ![Bedrooms](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/bedrooms.PNG?raw=true)

**Kitches above grade**
> ![Kitches](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/kitchen.PNG?raw=true)

**Toatal Rooms**
> ![Rooms](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/totalrooms.PNG?raw=true)

**Fire Places**
> ![Fireplace](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/Fireplaces.PNG?raw=true)

**Size of Garage in Car Capacity**
> ![Garage](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/gragesize.PNG?raw=true)

**Remodel Date**
> ![Remodelling](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/remodellingyear.PNG?raw=true)

**Original Construction Date**
> ![construction](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/yearbuilt.PNG?raw=true)

**Year Sold**
> ![sold](https://github.com/Pradnya1208/House-prices-prediction/blob/main/output/yearsold.PNG?raw=true)

### 2. Continuous Variables:



