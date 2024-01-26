import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pickle
from sklearn.model_selection import KFold

# %matplotlib inline

data = pd.read_csv("../data/kidney_disease.csv")
print(data.head())
print(data.info())
print(data.classification.unique())

data.classification = data.classification.replace("ckd\t", "ckd")
print(data.classification.unique())

data.drop("id", axis=1, inplace=True)
print(data.head())

data["classification"] = data["classification"].replace(["ckd", "notckd"], [1, 0])
print(data.head())

print(data.isnull().sum())

df = data.dropna(axis=0).copy()  # Create a copy of the DataFrame
print(f"Before dropping all NaN values: {data.shape}")
print(f"After dropping all NaN values: {df.shape}")
print(df.head())

df.index = range(0, len(df), 1)
print(df.head())

for i in df["wc"]:
    print(i)

df.loc[:, "wc"] = df["wc"].replace(["\t6200", "\t8400"], [6200, 8400])

for i in df["wc"]:
    print(i)

print(df.info())

df.loc[:, "pcv"] = df["pcv"].astype(int)
df.loc[:, "wc"] = df["wc"].astype(int)
df.loc[:, "rc"] = df["rc"].astype(float)

print(df.info())

object_dtypes = df.select_dtypes(include="object")
print(object_dtypes.head())

dictonary = {
    "rbc": {
        "abnormal": 1,
        "normal": 0,
    },
    "pc": {
        "abnormal": 1,
        "normal": 0,
    },
    "pcc": {
        "present": 1,
        "notpresent": 0,
    },
    "ba": {
        "notpresent": 0,
        "present": 1,
    },
    "htn": {
        "yes": 1,
        "no": 0,
    },
    "dm": {
        "yes": 1,
        "no": 0,
    },
    "cad": {
        "yes": 1,
        "no": 0,
    },
    "appet": {
        "good": 1,
        "poor": 0,
    },
    "pe": {
        "yes": 1,
        "no": 0,
    },
    "ane": {
        "yes": 1,
        "no": 0,
    },
}

df = df.replace(dictonary)

# Correcting the data type conversion part
df["rbc"] = df["rbc"].astype(int)
df["pc"] = df["pc"].astype(int)
df["pcc"] = df["pcc"].astype(int)
df["ba"] = df["ba"].astype(int)
df["htn"] = df["htn"].astype(int)
df["dm"] = df["dm"].astype(int)
df["cad"] = df["cad"].astype(int)
df.loc[:, "appet"] = df.loc[:, "appet"].astype(int)  # Corrected this line
df.loc[:, "pe"] = df.loc[:, "pe"].astype(int)  # Corrected this line
df.loc[:, "ane"] = df.loc[:, "ane"].astype(int)  # Corrected this line

print(df.head())

import seaborn as sns

plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=0.5)
print(df.corr())

X = df.drop(["classification", "sg", "appet", "rc", "pcv", "hemo", "sod"], axis=1)
y = df["classification"]
print(X.columns)

kf = KFold(n_splits=5, random_state=42, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf": [1, 2, 5, 10],
}

# Create a RandomForestClassifier
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print(f"Best parameters: {best_params}")

# Train the model using the best parameters
model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_features=best_params["max_features"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
)

model.fit(X_train, y_train)

# Evaluate the model
print(confusion_matrix(y_test, model.predict(X_test)))
print(f"Accuracy is {round(accuracy_score(y_test, model.predict(X_test))*100, 2)}%")

# Save the model
pickle.dump(model, open("../models/kidney.pkl", "wb"))
