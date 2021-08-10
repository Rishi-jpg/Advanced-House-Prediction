# Data Analysis Phase
# Main aim is to understand more about the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns',None)

dataset = pd.read_csv("data/train.csv")

# Print shape of dataset with rows and columns
print(dataset.shape)

# Print the top 5 records
h = dataset.head()

# In Data Analysis we will do
# 1) Missing value
# 2) All the numerical variables
# 3) Distribution of the numerical variables
# 4) Categorical Variable
# 5) Cardinaity of Categorical Variable
# 6) Outliers
# 7) Relationship between independent and dependent feature(Sale Price)

# MISSING VALUES 
# in feature engineering we will handle the misssing value but here we only analyse it
# here we will check the percentage of nan values present in each feature 
# 1-Step make the list of feature which have missing value

features_with_nan = [features for features in dataset.columns if dataset[features].isnull().sum()>1]

# 2-Step print the feature name and percentage of missing value

for feature in features_with_nan:
    print(feature, np.round(dataset[feature].isnull().mean(),4), '%missing value')
    # (feature name, getting mean till 4 decimal of isnull value(percentage))


# Since There are many missing values, we need to find the relationship between missing values and sales Price
for feature in features_with_nan :
    data = dataset.copy()
    
    #let's make a variable that indicates 1 if the observation was missing or 0 if it is present
    data[feature] = np.where(data[feature].isnull(),1,0)
    
    #let's calculate the mean SalePrice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()

# Here with the relation between the missing values and the dependent variable is clearly visible.So we need to replace this nan values with something meaningful which we will do in the feature engineering section

# From the above dataset some of the features like id is not required
print("Id of houses {}".format(len(dataset.Id)))    


# HOW MANY FEATURES ARE NUMERICAL VARIABLE
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtype != 'O']#'O' means object

print('Numbers of numerical variables: ', len(numerical_features))

nf = dataset[numerical_features].head()

#Temporal Variables

year_features = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

# Lets analyze the temporal datetime variable
# we will check whether there is a relation between year the house is sold and sales price

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Sale Price')
plt.title("House Price vs Year Sold")

print(year_features)

## Here we will compare the difference between All years feature with SalePrice
for feature in year_features:
    if feature != "YrSold":
        data = dataset.copy()
        data[feature] = data["YrSold"]-data[feature]
        
        plt.scatter(data[feature],data["SalePrice"])
        plt.xlabel(feature)
        plt.ylabel("Sale price")
        plt.show()
        

## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_features = [feature for feature in numerical_features if len(dataset[feature].unique()) < 25 and feature not in year_features + ['Id']]
for feature in discrete_features:
    data = dataset.copy()
    data.groupby(feature)["SalePrice"].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("Sales Price")
    plt.title(feature)
    plt.show()
    
continous_features = [feature for feature in numerical_features if feature not in discrete_features+year_features+['Id']]
for feature in continous_features:
    data=dataset.copy()
    data[feature].hist(bins=20)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()

#we will transform our continous data into log transformation
for feature in continous_features:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title("Log Transformation")
        plt.show()

# OUTLIERS
for feature in continous_features:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        
# Categorical Feature 
cf = [feature for feature in dataset.columns if data[feature].dtypes == 'O']

# Number of Category in each category feature
for feature in cf:
    print("The feature is {} and number of cat is {}".format(feature,len(data[feature].unique())))

# HIgh number of categorical feature should be handle 
# category f having small number of category can be one hot incoded
    
#Find out the relationship between categorical variable and dependent feature SalesPrice
for feature in cf:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    
    
