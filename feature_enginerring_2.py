import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#to visualise all the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

dataset = pd.read_csv('data/train.csv')

# MISSING VALUES
# 1) let us capture all the nan values
# 2) first lets handle categorical features which are missing

feature_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes == 'O']

for feature in feature_nan:
    print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))
    
# Replace missing value with a new label
# If you want to handle missing value in category feature so create a new category for them
def replace_cat_feature(dataset,feature_nan):
    data = dataset.copy()
    data[feature_nan] = data[feature_nan].fillna('Missing')
    return data

dataset = replace_cat_feature(dataset, feature_nan)

for feature in feature_nan:
    print("{} null value is {}".format(feature,dataset[feature].isnull().sum()))

# 3) Now check for numerical variables that contain missing values
numerical_with_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']

# we will print the numerical nan variable and percentage of missing values
for feature in numerical_with_nan:
    print("\n{} is {}%".format(feature,np.round(dataset[feature].isnull().mean(),4)))
    
# Note:- we saw that in explotery data analysis we have lots of outliers in numerical feature 
# so when you have outlier you should definitely go replacing the nan value with (median or mode)

for feature in numerical_with_nan:
    # we will replace by using median since there are outliers
    median_value = dataset[feature].median()
    
    #if we have nan value in numerical feature always create a new feature to capture nan value...if nan = 1 else 0
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)

for feature in numerical_with_nan:
    print("{} null value = {}".format(feature,dataset[feature].isnull().sum()))


# 4) Temporal Variable (Date time Variable)
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    dataset[feature]= dataset['YrSold']-dataset[feature]


# NOW handling skewed data
# log tranformation
numerical_f = [feature for feature in dataset.columns if dataset[feature].dtypes!='O']
for feature in numerical_f:
    dataset[feature] = np.log(dataset[feature])


# NOW handling rare categorical feature
# we will remove categorical variables that are present less than 1% of the observation
cf = [feature for feature in dataset.columns if dataset[feature].dtypes=='O']

for feature in cf:
    temp = dataset.groupby(feature)['SalePrice'].count()/len(dataset) # grouping by category that are present in category feature
    temp_df = temp[temp>0.1].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')


# FEATURE SCALING
# 1) MinMaxScaler (scale value between 0 to 1)
# 2) Standard Scaler 
feature_scale = [feature for feature in dataset.columns if feature not in ['Id','ScalePrice']]
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(dataset[feature_scale])

# scaler will give value in array form so we need to convert it into dataset form
# transform the train and test set, and add on the Id and SalePrice variables
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],
                    axis=1)


data.to_csv('X_train.csv',index=False)









