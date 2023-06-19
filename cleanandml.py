#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use("ggplot")
rcParams['figure.figsize'] = (12, 6)

#load data
salary_data = pd.read_csv("Salary_Data.csv")
salary_data

#fast checks
salary_data.info()

salary_data.describe()
salary_data.columns

#checking for na values
salary_data.isna()
salary_data.isna().sum()

#remove na values
salarydata_clean = salary_data.dropna()
salarydata_clean.isna().sum()

#check for duplicates
salarydata_clean.duplicated().sum()

salarydata_clean = salarydata_clean.drop_duplicates(keep=False)
salarydata_clean.duplicated().sum()

salarydata_clean.shape

# plots
sns.pairplot(salarydata_clean)

#correlation check
print(salarydata_clean.corr())

# decision tree, linear regression, and random forest

#imports
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics mo


#first splitting dataset
col_names = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']

X = salarydata_clean[col_names]
y = salarydata_clean['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# the decision tree algorithm was with very bad results, 14 and 12 percent accuracy

#linear regression  -->

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error
p1 = regressor.predict(X_test)
print(mean_absolute_error(y_test,p1))
print(r2_score(y_test,p1))


#decision tree regressor

from sklearn.tree import DecisionTreeRegressor 
m1 = DecisionTreeRegressor(random_state=0)
m1.fit(X_train,y_train)


p2 = m1.predict(X_test)
print(mean_absolute_error(y_test,p2))
print(r2_score(y_test,p2))

#random forest regressor

from sklearn.ensemble import RandomForestRegressor
m2 = RandomForestRegressor(n_estimators=100)
m2.fit(X_train,y_train)

p3 = m2.predict(X_test)
print(r2_score(y_test,p3))

