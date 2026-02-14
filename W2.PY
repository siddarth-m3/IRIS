import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("Iris.csv")

df.head()   # Show first 5 rows

df.info()   # Show dataset information

df["Species"].value_counts()   # Count of each species

df.describe()   # Statistical summary of dataset


# Drop unnecessary column
df = df.drop("Id", axis = 1)


# Pairplot to see relationship between all features
sns.pairplot(df)
plt.show()


# Remove Species column to calculate correlation
drop_species = df.drop("Species", axis = 1)

# Heatmap to see correlation between features
plt.figure(figsize = (10,6))
sns.heatmap(drop_species.corr(), annot = True)
plt.show()


# Scatter plot: Sepal Length vs Sepal Width
sns.scatterplot(x = df["SepalLengthCm"], y = df["SepalWidthCm"], hue = df["Species"])
plt.show()

# Scatter plot: Petal Length vs Petal Width
sns.scatterplot(x = df["PetalLengthCm"], y = df["PetalWidthCm"], hue = df["Species"])
plt.show()


# Convert Species (text) into numbers
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])

df.head()   # Check encoded values

df["Species"].value_counts()   # Check class counts again


# Boxplots to compare features across species
plt.figure(figsize =(10,6))

plt.subplot(1,2,1)
sns.boxplot(x = df["Species"], y = df["SepalLengthCm"])
plt.title("Sepal Length")

plt.subplot(1,2,2)
sns.boxplot(x = df["Species"], y = df["PetalLengthCm"])
plt.title("Petal Length")

plt.show()


# Separate features (X) and target (y)
X = df.drop("Species", axis = 1)
y = df["Species"]


# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)


# Standardize features (important for some models)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Gaussian Naive Bayes Model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train_scaled, y_train)   # Train model
y_pred_gnb = gnb.predict(X_test_scaled)   # Make predictions


# Evaluate Naive Bayes model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("accuracy score: " , accuracy_score(y_test,y_pred_gnb))
print("confusion matrix: \n" , confusion_matrix(y_test,y_pred_gnb))
print("classification report: " , classification_report(y_test,y_pred_gnb))



# Logistic Regression Model

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

logistic.fit(X_train_scaled ,y_train)   
y_pred_log = logistic.predict(X_test_scaled)   
y_pred_log

# Evaluate Logistic Regression
print("accuracy score: " , accuracy_score(y_test,y_pred_log))
print("confusion matrix: \n" , confusion_matrix(y_test,y_pred_log))
print("classification report: " , classification_report(y_test,y_pred_log))


import warnings
warnings.filterwarnings("ignore")  

# Define parameters to search
penalty = ["l1","l2","elasticnet"]
c_values = [100, 10, 1, 0.1, 0.01]
solver = ["lbfgs", "liblinear","newton-cg","newton-cholesky","sag","saga"]
params = dict(penalty=penalty, C=c_values, solver=solver)

from sklearn.model_selection import GridSearchCV, StratifiedKFold
cv = StratifiedKFold()   # Cross-validation with balanced classes

grid = GridSearchCV(estimator = logistic, param_grid = params, cv = cv, scoring = "accuracy", n_jobs = -1)
grid  
grid.fit(X_train_scaled, y_train)   # Train with best parameters
y_pred_grid = grid.predict(X_test_scaled)   # Predict using best model
y_pred_grid

# Evaluate tuned model
print("accuracy score: " , accuracy_score(y_test,y_pred_grid))
print("confusion matrix: \n" , confusion_matrix(y_test,y_pred_grid))
print("classification report: " , classification_report(y_test,y_pred_grid))

grid.best_params_   # Show best parameters
grid.best_score_    # Show best cross-validation score



# Support Vector Machine (SVM)

from sklearn.svm import SVC

svc = SVC()

# Parameters for SVM tuning
param_grid = { 
    "kernel":["rbf","linear","poly","sigmoid"], 
    "C":[0.1,1,10,100,1000], 
    "gamma":["auto","scale"]
}

grid_svc = GridSearchCV(estimator = svc, param_grid = param_grid, cv = 5)

grid_svc.fit(X_train_scaled, y_train)   # Train SVM

y_pred_grid_svc = grid_svc.predict(X_test_scaled)   

y_pred_grid_svc

print("accuracy score: " , accuracy_score(y_test,y_pred_grid_svc))
print("confusion matrix: \n" , confusion_matrix(y_test,y_pred_grid_svc))
print("classification report: " , classification_report(y_test,y_pred_grid_svc))

grid_svc.best_params_   # Best SVM parameters
grid_svc.best_score_    # Best SVM score
